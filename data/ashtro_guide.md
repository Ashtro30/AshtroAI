# Ashtro-AI Quick Guide

Ashtro is an autonomous assistant that follows a Reason -> Plan -> Act -> Verify loop.

## What Ashtro Can Do

- **Search documents:** answers questions from the markdown and text files in the `data/` folder using semantic retrieval.
- **Run Python:** performs exact maths and data analysis, including loading CSV files from `data/` with `load_csv`.
- **Write reports:** saves Markdown files into the `outputs/` folder.
- **Plan tasks:** breaks goals into tracked tasks with statuses (pending, in_progress, done, blocked).

## Adding Knowledge

Drop `.md` or `.txt` files into `data/` (or upload them in the Knowledge tab of the app) and click Re-index. Unchanged files are skipped on later runs.

## Memory

Ashtro keeps a short rolling window of the current conversation and stores summaries of successful past tasks in a persistent vector store, so similar future requests can reuse earlier context.
