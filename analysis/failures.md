# Failure analysis (merged models)

Models: merged_linear, merged_ties, mopd. Source: `results/<model>/<benchmark>/` raw per-item files; full list in `analysis/failures.jsonl`. Categories are assigned by the rules in `scripts/analyze_failures.py`; sampled cases below are stratified per model x benchmark x category (seed 42).

## Category counts

| model | benchmark | category | count |
|---|---|---|---|
| merged_linear | asi | missing_outputs | 54 |
| merged_linear | asi | wrong_values | 4 |
| merged_linear | asi | scorer_timeout | 2 |
| merged_linear | hle | wrong_answer | 330 |
| merged_linear | hle | wrong_choice | 115 |
| merged_linear | hle | (of which raw local judge said correct) | 32 |
| merged_linear | hle | template_echo | 27 |
| merged_linear | hle | (of which merge_regression vs base) | 13 |
| merged_linear | hle | no_answer_extracted | 3 |
| merged_linear | hle | domain_format_leak | 1 |
| merged_linear | swe | no_patch | 5 |
| merged_linear | swe | env_tool_failure | 1 |
| merged_linear | terminal | task_not_achieved | 51 |
| merged_linear | terminal | timeout | 22 |
| merged_ties | asi | missing_outputs | 55 |
| merged_ties | asi | wrong_values | 3 |
| merged_ties | asi | scorer_timeout | 2 |
| merged_ties | hle | domain_format_leak | 198 |
| merged_ties | hle | wrong_answer | 175 |
| merged_ties | hle | wrong_choice | 92 |
| merged_ties | hle | (of which raw local judge said correct) | 17 |
| merged_ties | hle | no_answer_extracted | 14 |
| merged_ties | hle | (of which merge_regression vs base) | 13 |
| merged_ties | hle | template_echo | 4 |
| merged_ties | swe | no_patch | 5 |
| merged_ties | swe | env_tool_failure | 1 |
| merged_ties | terminal | task_not_achieved | 46 |
| merged_ties | terminal | timeout | 26 |
| mopd | asi | missing_outputs | 56 |
| mopd | asi | wrong_values | 4 |
| mopd | hle | wrong_answer | 359 |
| mopd | hle | wrong_choice | 119 |
| mopd | hle | (of which merge_regression vs base) | 11 |
| mopd | hle | no_answer_extracted | 5 |
| mopd | hle | (of which raw local judge said correct) | 3 |
| mopd | swe | no_patch | 4 |
| mopd | swe | env_tool_failure | 1 |
| mopd | swe | patch_fails_tests | 1 |
| mopd | terminal | task_not_achieved | 47 |
| mopd | terminal | timeout | 27 |

## Sampled cases (63)

1. **[merged_linear / asi] `computer_science.deployment_prediction_sets__seed31415`** — *missing_outputs*  
   score: 0.0; gate_messages: File not found: results/prediction_sets.csv; File not found: results/policy_parameters.csv; File not found: results/audit_summary.csv; File not found: results/s
2. **[merged_linear / asi] `astronomy.planet_activity_sep__seed31415`** — *missing_outputs*  
   score: 0.0; gate_messages: File not found: signal_summary.csv; File not found: target_predictions.csv
3. **[merged_linear / asi] `computer_science.max3sat_assignment_optimization__seed31415`** — *scorer_timeout*  
   score: 0.0; gate_messages: scorer_timeout (300s)
4. **[merged_linear / asi] `computer_science.euclidean_tsp_tour_optimization__seed31415`** — *scorer_timeout*  
   score: 0.0; gate_messages: scorer_timeout (300s)
5. **[merged_linear / asi] `math.sigma_linear_diophantine__seed31415`** — *wrong_values*  
   score: 0.0; gate_messages: generated artifacts or executable checks failed
6. **[merged_linear / asi] `math.levin_context_grid_search__seed31415`** — *wrong_values*  
   score: 0.0
7. **[merged_linear / hle] `66eeea0187ec4a6a04092e78`** — *domain_format_leak*  
   merge_regression: False; judge_false_positive: False; category_raw: Computer Science/AI; question/gold: [redacted: HLE test content]
8. **[merged_linear / hle] `671aaca917615f09eab6fe37`** — *no_answer_extracted*  
   merge_regression: False; judge_false_positive: False; category_raw: Math; question/gold: [redacted: HLE test content]
9. **[merged_linear / hle] `66e926cd0a94aa76a5ed8df8`** — *no_answer_extracted*  
   merge_regression: False; judge_false_positive: False; category_raw: Other; question/gold: [redacted: HLE test content]
10. **[merged_linear / hle] `6706671e0ae45a43cc2b8e8f`** — *template_echo*  
   merge_regression: True; judge_false_positive: False; model_answer: {your chosen answer}; category_raw: Math; question/gold: [redacted: HLE test content]
11. **[merged_linear / hle] `66fc2559876b283872e52acc`** — *template_echo*  
   merge_regression: False; judge_false_positive: True; model_answer: {your chosen answer}; category_raw: Math; question/gold: [redacted: HLE test content]
12. **[merged_linear / hle] `6747df97a3299f69eec20468`** — *wrong_answer*  
   merge_regression: True; judge_false_positive: False; model_answer: $1 - 2^{4048}$; category_raw: Math; question/gold: [redacted: HLE test content]
13. **[merged_linear / hle] `6725292085b48a76ea1b5709`** — *wrong_answer*  
   merge_regression: True; judge_false_positive: False; model_answer: 5; category_raw: Math; question/gold: [redacted: HLE test content]
14. **[merged_linear / hle] `671e3d672637abea9c147ba1`** — *wrong_choice*  
   merge_regression: True; judge_false_positive: False; 'b' - 'the particle is in the interval [-1, 1]', 'c' ; model_answer: B; category_raw: Physics; question/gold: [redacted: HLE test content]
15. **[merged_linear / hle] `671ac641136758472327f851`** — *wrong_choice*  
   merge_regression: True; judge_false_positive: False; model_answer: A; category_raw: Biology/Medicine; question/gold: [redacted: HLE test content]
16. **[merged_linear / swe] `swebench-pro/instance_element-hq__element-web-772df3021201d9c73835a626`** — *env_tool_failure*  
   exception: RuntimeError; patch_bytes: 0
17. **[merged_linear / swe] `swebench-pro/instance_future-architect__vuls-ef2be3d6ea4c0a13674aaab08`** — *no_patch*  
   patch_bytes: 0
18. **[merged_linear / swe] `swebench-pro/instance_ansible__ansible-d33bedc48fdd933b5abd65a77c08187`** — *no_patch*  
   patch_bytes: 0
19. **[merged_linear / terminal] `break-filter-js-from-html`** — *task_not_achieved*  
   
20. **[merged_linear / terminal] `train-fasttext`** — *task_not_achieved*  
   
21. **[merged_linear / terminal] `constraints-scheduling`** — *timeout*  
   exception: AgentTimeoutError
22. **[merged_linear / terminal] `dna-insert`** — *timeout*  
   exception: AgentTimeoutError
23. **[merged_ties / asi] `computer_science.stable_latent_state_fitting__seed31415`** — *missing_outputs*  
   score: 0.0; gate_messages: File not found: results/candidate_summary.csv; File not found: results/matrix_a.csv; File not found: results/matrix_b.csv; File not found: results/vector_a.csv;
24. **[merged_ties / asi] `math.nnls_modulus_deblur__seed31415`** — *missing_outputs*  
   score: 0.0; gate_messages: File not found: reconstruction.npy; File not found: reconstruction.png; File not found: iteration_log.csv; File not found: metrics.json | Gate FAIL: reconstruct
25. **[merged_ties / asi] `computer_science.euclidean_tsp_tour_optimization__seed31415`** — *scorer_timeout*  
   score: 0.0; gate_messages: scorer_timeout (300s)
26. **[merged_ties / asi] `computer_science.max3sat_assignment_optimization__seed31415`** — *scorer_timeout*  
   score: 0.0; gate_messages: scorer_timeout (300s)
27. **[merged_ties / asi] `math.sigma_linear_diophantine__seed31415`** — *wrong_values*  
   score: 0.0; gate_messages: generated artifacts or executable checks failed
28. **[merged_ties / asi] `math.ucb_q_learning_regret__seed31415`** — *wrong_values*  
   score: 0.0
29. **[merged_ties / hle] `671e3d672637abea9c147ba1`** — *domain_format_leak*  
   merge_regression: True; judge_false_positive: False; 'b' - 'the particle is in the interval [-1, 1]', 'c' ; category_raw: Physics; question/gold: [redacted: HLE test content]
30. **[merged_ties / hle] `670dc75dcfd3fc87a109a929`** — *domain_format_leak*  
   merge_regression: True; judge_false_positive: False; category_raw: Math; question/gold: [redacted: HLE test content]
31. **[merged_ties / hle] `6725292085b48a76ea1b5709`** — *no_answer_extracted*  
   merge_regression: True; judge_false_positive: True; category_raw: Math; question/gold: [redacted: HLE test content]
32. **[merged_ties / hle] `673245364fe7531d658b2750`** — *no_answer_extracted*  
   merge_regression: False; judge_false_positive: False; category_raw: Math; question/gold: [redacted: HLE test content]
33. **[merged_ties / hle] `67208d2f5241ba1dad593bf5`** — *template_echo*  
   merge_regression: False; judge_false_positive: True; model_answer: {your chosen answer}; category_raw: Math; question/gold: [redacted: HLE test content]
34. **[merged_ties / hle] `671ca6ce84c08d69293fb129`** — *template_echo*  
   merge_regression: False; judge_false_positive: True; model_answer: {your chosen answer}; category_raw: Math; question/gold: [redacted: HLE test content]
35. **[merged_ties / hle] `673a73d1addb9293360a5d07`** — *wrong_answer*  
   merge_regression: True; judge_false_positive: False; model_answer: No; category_raw: Math; question/gold: [redacted: HLE test content]
36. **[merged_ties / hle] `6706671e0ae45a43cc2b8e8f`** — *wrong_answer*  
   merge_regression: True; judge_false_positive: False; model_answer: 4096; category_raw: Math; question/gold: [redacted: HLE test content]
37. **[merged_ties / hle] `67254a56046774791d0f1a41`** — *wrong_choice*  
   merge_regression: True; judge_false_positive: False; model_answer: A. Ebstein anomaly; category_raw: Biology/Medicine; question/gold: [redacted: HLE test content]
38. **[merged_ties / hle] `67313652f659ba7b3fd1fe40`** — *wrong_choice*  
   merge_regression: True; judge_false_positive: False; model_answer: B; category_raw: Humanities/Social Science; question/gold: [redacted: HLE test content]
39. **[merged_ties / swe] `swebench-pro/instance_element-hq__element-web-772df3021201d9c73835a626`** — *env_tool_failure*  
   exception: RuntimeError; patch_bytes: 0
40. **[merged_ties / swe] `swebench-pro/instance_ansible__ansible-d33bedc48fdd933b5abd65a77c08187`** — *no_patch*  
   patch_bytes: 0
41. **[merged_ties / swe] `swebench-pro/instance_future-architect__vuls-ef2be3d6ea4c0a13674aaab08`** — *no_patch*  
   patch_bytes: 0
42. **[merged_ties / terminal] `cobol-modernization`** — *task_not_achieved*  
   
43. **[merged_ties / terminal] `dna-assembly`** — *task_not_achieved*  
   
44. **[merged_ties / terminal] `adaptive-rejection-sampler`** — *timeout*  
   exception: AgentTimeoutError
45. **[merged_ties / terminal] `extract-moves-from-video`** — *timeout*  
   exception: AgentTimeoutError
46. **[mopd / asi] `biology.contextual_dna_regime_mapping__seed31415`** — *missing_outputs*  
   score: 0.0; gate_messages: File not found: predicted_regions.csv; File not found: state_paths.npz; File not found: posterior_probs.npz; File not found: fit_diagnostics.json; File not foun
47. **[mopd / asi] `biostat.perturbation_convergence_testing__seed31415`** — *missing_outputs*  
   score: 0.0; gate_messages: File not found: results/case_0/convergence_summary.json; File not found: results/case_0/convergence_pvalues.csv; File not found: results/case_0/convergence_deci
48. **[mopd / asi] `math.sigma_linear_diophantine__seed31415`** — *wrong_values*  
   score: 0.0; gate_messages: generated artifacts or executable checks failed
49. **[mopd / asi] `math.ucb_q_learning_regret__seed31415`** — *wrong_values*  
   score: 0.0
50. **[mopd / hle] `66fde5a291664bba6e391cb5`** — *no_answer_extracted*  
   merge_regression: False; judge_false_positive: False; category_raw: Math; question/gold: [redacted: HLE test content]
51. **[mopd / hle] `67073a60c033e34a5d675f2e`** — *no_answer_extracted*  
   merge_regression: False; judge_false_positive: False; category_raw: Math; question/gold: [redacted: HLE test content]
52. **[mopd / hle] `6706671e0ae45a43cc2b8e8f`** — *wrong_answer*  
   merge_regression: True; judge_false_positive: False; model_answer: 4; category_raw: Math; question/gold: [redacted: HLE test content]
53. **[mopd / hle] `66fb417395a8e2fc57e479d7`** — *wrong_answer*  
   merge_regression: True; judge_false_positive: False; model_answer: 2000; category_raw: Physics; question/gold: [redacted: HLE test content]
54. **[mopd / hle] `67254a56046774791d0f1a41`** — *wrong_choice*  
   merge_regression: True; judge_false_positive: False; model_answer: B. Patent ductus arteriosus; category_raw: Biology/Medicine; question/gold: [redacted: HLE test content]
55. **[mopd / hle] `671a58056e1db673f77905cd`** — *wrong_choice*  
   merge_regression: True; judge_false_positive: False; model_answer: E; category_raw: Humanities/Social Science; question/gold: [redacted: HLE test content]
56. **[mopd / swe] `swebench-pro/instance_element-hq__element-web-772df3021201d9c73835a626`** — *env_tool_failure*  
   exception: RuntimeError; patch_bytes: 0
57. **[mopd / swe] `swebench-pro/instance_internetarchive__openlibrary-322d7a46cdc965bfabb`** — *no_patch*  
   patch_bytes: 0
58. **[mopd / swe] `swebench-pro/instance_future-architect__vuls-ef2be3d6ea4c0a13674aaab08`** — *no_patch*  
   patch_bytes: 0
59. **[mopd / swe] `swebench-pro/instance_flipt-io__flipt-e2bd19dafa7166c96b082fb2a59eb54b`** — *patch_fails_tests*  
   patch_bytes: 6067
60. **[mopd / terminal] `mcmc-sampling-stan`** — *task_not_achieved*  
   
61. **[mopd / terminal] `git-leak-recovery`** — *task_not_achieved*  
   
62. **[mopd / terminal] `cobol-modernization`** — *timeout*  
   exception: AgentTimeoutError
63. **[mopd / terminal] `crack-7z-hash`** — *timeout*  
   exception: AgentTimeoutError
