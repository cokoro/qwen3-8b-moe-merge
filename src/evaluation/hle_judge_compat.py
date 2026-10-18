"""Run the official HLE judge (third_party/hle/hle_eval/run_judge_results.py) unchanged,
with two runtime adaptations for OpenAI-compatible judges that are not o3-mini:

1. JSON hint: some providers (ModelScope / DashScope) implement structured output as
   ``response_format=json_object``, which requires the word "json" in the messages.
   We append one sentence asking for the same fields as JSON. The judging
   instructions themselves are untouched.
2. Retries: each judge call is retried with exponential backoff (rate limits), and
   the whole judge pass is repeated until every prediction is judged or
   ``--passes`` is exhausted (the official script resumes from its own cache).

    python -m src.evaluation.hle_judge_compat --dataset DIR --predictions FILE --judge MODEL
"""

import argparse
import asyncio
import importlib.util
import json
import os
import random
from pathlib import Path

OFFICIAL = Path(__file__).resolve().parents[2] / "third_party/hle/hle_eval/run_judge_results.py"
JSON_HINT = (
    "\n\nReturn your judgement as a JSON object with the fields: extracted_final_answer, "
    'reasoning, correct ("yes" or "no"), confidence (integer 0-100), strict (true).'
)


def load_official():
    spec = importlib.util.spec_from_file_location("hle_run_judge_results", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=int(os.environ.get("HLE_JUDGE_RETRIES", "5")))
    parser.add_argument("--passes", type=int, default=int(os.environ.get("HLE_JUDGE_PASSES", "3")))
    parser.add_argument("--no_json_hint", action="store_true")
    parser.add_argument("--min_interval", type=float, default=float(os.environ.get("HLE_JUDGE_MIN_INTERVAL", "1.5")),
                        help="seconds between request starts (providers reject bursty clients)")
    args = parser.parse_args()

    hle = load_official()
    if os.environ.get("HLE_JUDGE_TIMEOUT"):  # local judges can loop to max tokens; cap each call
        from openai import AsyncOpenAI
        hle.client = AsyncOpenAI(timeout=float(os.environ["HLE_JUDGE_TIMEOUT"]), max_retries=0)
    if not args.no_json_hint:
        hle.JUDGE_PROMPT = hle.JUDGE_PROMPT + JSON_HINT
    hle.args = args

    original = hle.extract_answer
    pace = {"lock": None, "next": 0.0}

    async def paced():
        if pace["lock"] is None:
            pace["lock"] = asyncio.Lock()
        async with pace["lock"]:
            loop = asyncio.get_running_loop()
            wait = pace["next"] - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            pace["next"] = loop.time() + args.min_interval

    async def extract_with_retry(question, correct_answer, response):
        for attempt in range(args.retries):
            await paced()
            result = await original(question, correct_answer, response)
            if result is not None:
                return result
            await asyncio.sleep(min(60, 2 ** attempt * 3) + random.random())
        return None

    hle.extract_answer = extract_with_retry

    predictions = json.loads(Path(args.predictions).read_text())
    judged_path = Path(f"judged_{os.path.basename(args.predictions)}.json")
    for p in range(args.passes):
        try:
            hle.main(args)
        except IndexError as exc:  # official calib_err needs >= 100 judged items; judged file is already saved
            print(f"official metrics skipped ({exc})")
        judged = json.loads(judged_path.read_text()) if judged_path.exists() else {}
        missing = [k for k in predictions if k not in judged]
        print(f"judge pass {p + 1}: {len(judged)}/{len(predictions)} judged, {len(missing)} missing")
        if not missing:
            break


if __name__ == "__main__":
    main()
