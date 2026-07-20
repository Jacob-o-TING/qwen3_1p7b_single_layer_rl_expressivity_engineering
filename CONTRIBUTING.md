# Contributing

## Development setup

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests
```

GPU-specific tests require the pinned runtime documented in the corresponding
experiment record and are not part of the lightweight CI job.

## Research records

- Keep experiment plans and records in a natural Chinese-English mixed style
  when that preserves precise technical terminology.
- Use neutral scientific prose. Keep private names, credentials, personal
  notes, and ephemeral SSH endpoints outside durable reports.
- Preserve negative results and protocol changes. Do not silently rewrite a
  historical score after changing prompts, parsers, caps, or backends.
- Commit compact metrics and manifests only. Keep datasets, checkpoints, raw
  generations, logs, and chat transcripts outside Git.
- Carry every unresolved item from the canonical PENDING registry into new
  experiment plans.

## Pull requests

Keep changes scoped and include tests proportional to the behavioral risk.
