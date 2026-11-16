# Publication Policy

Publish source code, synthetic tests, model metadata and aggregate research reports only. Do not publish sportsbook odds, betting slips, selections, individual forecasts, shadow probabilities, settlement records or the accounting ledger. Files under `.private/` are local backups and must never be pushed.

## Local Guards

Install the committed hooks in every clone:

```sh
git config core.hooksPath .githooks
```

The pre-commit hook checks the entire index, including forced additions. The pre-push hook checks every reachable commit in the branches/tags being pushed, not just the latest tree. It rejects private paths and recognizable betting records copied into other JSON, CSV or Markdown files. Only source and reviewed aggregate report formats may be published from `outputs/`.

Run the checks manually:

```sh
python3 scripts/check_publication.py --staged
python3 scripts/check_publication.py --history HEAD
```

GitHub Actions repeats the history check on pushes and pull requests. Hooks can be bypassed by an administrator, and CI runs after a push; neither is a substitute for reviewing staged content. Do not use `--no-verify` or force-add private files.

## Rewritten History

The privacy cleanup rewrites the published history. Other clones must be replaced or carefully rebased onto the new history. Never merge an old branch or push an old tag: either can restore deleted records. Local frozen betting files are preserved and ignored, not deleted or rescored.

GitHub may retain unreachable objects and cached old commit views after a force push. GitHub Support controls server-side garbage collection and cached-view removal. Existing third-party clones cannot be remotely erased.
