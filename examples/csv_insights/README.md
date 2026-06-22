# CSV insights pipeline — a real `respawn.recover()` example

A dependency-free, fully offline showcase of respawn in its validated regime:
a **stateful pipeline where a silent upstream fault corrupts the result**, and
re-running the final step can't fix it.

The pipeline (`load → parse_amount → filter → aggregate → summarize`) computes
total revenue and the top category from a messy sales CSV. The default
`parse_amount` strategy can't read amounts like `"$1,200.00"` and silently zeroes
them — so the run reports **$2,056** instead of the true **$16,117**, and even
names the wrong top category. No crash, just a confidently wrong insight.

```
python examples/csv_insights/demo.py
```

What it shows:

- **retry-at-crash** (re-run the `summarize` step) → still $2,056. The cause is
  upstream; re-running the summary can't un-corrupt the data.
- **respawn** → a heuristic attributor flags `parse_amount` (where data quality
  collapsed), `recover()` re-enters there, switches to the robust parser, and
  recomputes forward → correct **$16,117**, right top category.

This is the pattern respawn is for: each step has a ladder of strategies, the
pipeline checkpoints state so it can roll back, and `recover()` drives
re-enter-at-the-cause + retry-differently. See [`pipeline.py`](pipeline.py) and
[`demo.py`](demo.py).