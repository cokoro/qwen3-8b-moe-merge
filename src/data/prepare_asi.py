"""Prepare ASI (agent/tool-use) expert training data.

NOTE: This module synthesizes agent-style planning examples from hand-authored
templates and optionally a public dataset (tau/commonsense_qa used as a planning
proxy). The real ASI-Bench test set must remain completely untouched until final
evaluation. Do NOT load or inspect ASI-Bench test instances here.
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset

from src.data.common import write_splits


# Hand-authored agent/tool-use planning templates.
# Each entry has a task description and a multi-step plan response.
TEMPLATES: List[Dict] = [
    {
        "task": "Search the web for the latest research on transformer attention mechanisms, then summarize the top 3 papers and save the summary to a file.",
        "plan": (
            "Step 1: Use `web_search(query='latest transformer attention mechanisms research 2024')` to retrieve recent papers.\n"
            "Step 2: For each of the top 3 results, use `fetch_url(url=<result_url>)` to retrieve the full abstract.\n"
            "Step 3: Use `summarize(text=<abstract>)` to produce a concise summary for each paper.\n"
            "Step 4: Use `write_file(path='attention_summary.txt', content=<combined_summaries>)` to persist the output."
        ),
    },
    {
        "task": "Check whether a remote REST API endpoint is healthy, log the response time, and send an alert if it exceeds 500ms.",
        "plan": (
            "Step 1: Use `http_get(url=<endpoint>, timeout=5)` and record the wall-clock duration.\n"
            "Step 2: Use `log_metric(name='api_response_ms', value=<duration_ms>)` to record the measurement.\n"
            "Step 3: If `duration_ms > 500`, use `send_alert(channel='ops', message='API latency {duration_ms}ms exceeds threshold')` to notify the team."
        ),
    },
    {
        "task": "Clone a GitHub repository, run its test suite, and open a bug report if any tests fail.",
        "plan": (
            "Step 1: Use `shell(cmd='git clone <repo_url> /tmp/repo')` to clone the repository.\n"
            "Step 2: Use `shell(cmd='cd /tmp/repo && python -m pytest --tb=short 2>&1')` to run the tests.\n"
            "Step 3: Parse the output; if failures are detected, use `create_github_issue(repo=<repo>, title='CI failure', body=<test_output>)` to file a bug report."
        ),
    },
    {
        "task": "Read a CSV file of customer orders, compute the total revenue per product, and write a sorted report to a new CSV.",
        "plan": (
            "Step 1: Use `read_file(path='orders.csv')` to load the raw data.\n"
            "Step 2: Parse each row, grouping by product and accumulating `quantity * unit_price`.\n"
            "Step 3: Sort the aggregated results by total_revenue descending.\n"
            "Step 4: Use `write_file(path='revenue_report.csv', content=<sorted_rows>)` to persist the output."
        ),
    },
    {
        "task": "Monitor a directory for new PDF files, extract their text, and index the content into a search database.",
        "plan": (
            "Step 1: Use `watch_directory(path='/inbox', pattern='*.pdf')` to register a file-arrival hook.\n"
            "Step 2: On each new file event, use `extract_text(file=<pdf_path>)` to get plain text.\n"
            "Step 3: Use `index_document(db='search_db', doc_id=<filename>, text=<extracted_text>)` to add the document to the search index."
        ),
    },
    {
        "task": "Translate a batch of product descriptions from English to French and Spanish, then upload the results to an S3 bucket.",
        "plan": (
            "Step 1: Use `read_file(path='products.jsonl')` to load the descriptions.\n"
            "Step 2: For each item, call `translate(text=<description>, target_lang='fr')` and `translate(text=<description>, target_lang='es')` in parallel.\n"
            "Step 3: Merge results into `products_translated.jsonl`.\n"
            "Step 4: Use `s3_upload(bucket='my-bucket', key='products_translated.jsonl', file='products_translated.jsonl')` to upload."
        ),
    },
    {
        "task": "Given a user question, retrieve relevant documents from a vector store, rerank them, and generate a grounded answer.",
        "plan": (
            "Step 1: Use `embed(text=<user_question>)` to produce a query vector.\n"
            "Step 2: Use `vector_search(index='docs', vector=<query_vec>, top_k=20)` to retrieve candidates.\n"
            "Step 3: Use `rerank(query=<user_question>, docs=<candidates>, top_n=5)` to select the most relevant passages.\n"
            "Step 4: Use `generate(prompt=<user_question>, context=<top_passages>)` to produce the final answer."
        ),
    },
    {
        "task": "Scrape product prices from three competitor websites and store the data in a database table.",
        "plan": (
            "Step 1: For each site in ['site_a.com', 'site_b.com', 'site_c.com'], use `fetch_url(url=<site>/products)` to get HTML.\n"
            "Step 2: Use `parse_html(html=<html>, selector='.price')` to extract price elements.\n"
            "Step 3: Normalize the price strings to floats.\n"
            "Step 4: Use `db_insert(table='competitor_prices', rows=<parsed_rows>)` to persist each batch."
        ),
    },
    {
        "task": "Run a nightly data pipeline: extract from Postgres, transform with dbt, load into BigQuery, and send a Slack summary.",
        "plan": (
            "Step 1: Use `db_query(conn='postgres://...', sql='SELECT * FROM orders WHERE date = CURRENT_DATE')` to extract today's data.\n"
            "Step 2: Use `shell(cmd='dbt run --select staging.orders')` to apply transformations.\n"
            "Step 3: Use `bq_load(dataset='warehouse', table='orders', source='dbt_output/')` to load to BigQuery.\n"
            "Step 4: Use `slack_post(channel='#data-ops', message='Nightly pipeline completed. Rows loaded: <count>')` to notify the team."
        ),
    },
    {
        "task": "Given a GitHub PR URL, review the diff for security issues, post inline comments, and update the PR status check.",
        "plan": (
            "Step 1: Use `fetch_pr_diff(url=<pr_url>)` to retrieve the unified diff.\n"
            "Step 2: Analyze each changed file for common vulnerability patterns (SQL injection, hardcoded secrets, etc.).\n"
            "Step 3: For each finding, use `post_review_comment(pr=<pr_url>, file=<file>, line=<line>, body=<finding>)` to add inline feedback.\n"
            "Step 4: Use `set_commit_status(sha=<head_sha>, state='failure' if findings else 'success', description=<summary>)` to update the check."
        ),
    },
]


def make_sample_from_template(template: Dict, idx: int) -> Dict:
    """Convert a hand-authored template into a training sample."""
    task = template["task"]
    plan = template["plan"]
    sample_id = f"asi_template_{idx:04d}_{hashlib.md5(task.encode()).hexdigest()[:6]}"
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": f"Please plan the following multi-step task:\n\n{task}"},
            {"role": "assistant", "content": plan},
        ],
        "source": "asi_handauthored",
        "split": "train",
    }


def collect_commonsense_planning_data(max_samples: int) -> List[Dict]:
    """Wrap commonsense_qa items as simple planning tasks.

    tau/commonsense_qa is used as a lightweight extension; it is NOT part of
    ASI-Bench. The real ASI-Bench test set must stay untouched until evaluation.
    """
    examples = []
    try:
        dataset = load_dataset("tau/commonsense_qa", split=f"train[:{max_samples}]")
        for item in dataset:
            question = item.get("question", "")
            choices = item.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if not question or not texts:
                continue
            options_str = "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
            task_prompt = (
                f"You need to answer the following question step-by-step, reasoning through each option:\n\n"
                f"{question}\n\nOptions:\n{options_str}"
            )
            answer_key = item.get("answerKey", "")
            answer_idx = labels.index(answer_key) if answer_key in labels else 0
            answer_text = texts[answer_idx] if texts else ""
            plan = (
                f"Step 1: Read the question carefully and identify the key concept being tested.\n"
                f"Step 2: Evaluate each option against background knowledge.\n"
                f"Step 3: Eliminate options that are clearly incorrect.\n"
                f"Step 4: Select the best answer.\n\n"
                f"Answer: {answer_key}. {answer_text}"
            )
            sample_id = f"asi_csqa_{hashlib.md5(question.encode()).hexdigest()[:8]}"
            examples.append({
                "id": sample_id,
                "messages": [
                    {"role": "user", "content": task_prompt},
                    {"role": "assistant", "content": plan},
                ],
                "source": "asi_commonsense_qa_proxy",
                "split": "train",
            })
    except Exception as exc:
        print(f"Warning: could not load tau/commonsense_qa: {exc}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare ASI expert training data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=4000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    samples: List[Dict] = []

    print("Building hand-authored agent planning templates…")
    for idx, tmpl in enumerate(TEMPLATES):
        samples.append(make_sample_from_template(tmpl, idx))

    remaining = args.max_samples - len(samples)
    if remaining > 0:
        print(f"Collecting commonsense_qa proxy data (up to {remaining} items)…")
        samples.extend(collect_commonsense_planning_data(remaining))

    write_splits(samples, output_path, args.max_samples)

if __name__ == "__main__":
    main()
