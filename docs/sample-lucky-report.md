> Sample with invented questions, to show the layout. Not a result.

# Lucky-pass report: run `sample`

A **lucky pass** is an answer judged correct although the gold passage was not retrieved.

- Questions: **8**
- Answer accuracy: **75.0%**
- True-pass rate (correct and gold retrieved): **37.5%**
- **Overstatement** (accuracy minus true-pass rate = lucky-pass rate): **37.5%** (95% bootstrap CI 12.5% to 75.0%, n = 8)
- Lucky passes: 3: 2 memory lucky (correct with no context), 1 guess lucky (wrong with no context), 0 not checked

## Lucky passes

| Question | Expected | RAG answer | Closed-book answer | Kind |
|---|---|---|---|---|
| What is the capital of France? | Paris | Paris | Paris | memory_lucky |
| Who wrote Pride and Prejudice? | Jane Austen | Jane Austen | Jane Austen | memory_lucky |
| When is the survey office's annual report due? | March 31 | March 31 | June 30 | guess_lucky |

The full per-question table is in `lucky.csv` next to this file.
