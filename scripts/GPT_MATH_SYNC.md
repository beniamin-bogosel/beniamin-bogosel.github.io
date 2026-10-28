# GPT Math updates

`sync_gpt_math.py` reads the results table in the
[GPT_Math README](https://github.com/beniamin-bogosel/GPT_Math#readme)
and updates `_data/gpt_math.json`. `GPT_Math.html` renders this JSON at build time.
The script uses Python's standard library; no packages or additional secrets are needed.

The **Sync GPT Math** GitHub Actions workflow checks daily at 05:23 UTC.
It commits changed JSON to `main` and explicitly starts the Pages build and
deployment. When the JSON is unchanged, it does neither. The workflow becomes
active after these files are pushed to `main`; it can also be started from
**Actions → Sync GPT Math → Run workflow**. GitHub's scheduled runs may be delayed;
[GitHub also disables schedules after 60 days without repository activity](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

For a local refresh:

```sh
python3 scripts/sync_gpt_math.py
```

To check without writing, add `--check` (exit status 1 means changes or an error).
For an offline check, use `--readme /path/to/README.md`; `--output` selects a
different JSON destination.

The upstream table controls titles, PDF links, result-directory links, reference
notes, and entry order. Entries removed from a valid, nonempty table are removed from the JSON.
Existing authors, years, abstracts, revision dates, and AI credits remain editable
in JSON and are preserved when either the PDF or a nonempty result-directory link matches.
New entries default to Beniamin Bogosel and the current UTC year; edit the year
for older notes and add optional metadata as needed. If both links change at
once, the row is treated as a new entry. No abstract is invented for a new row.

Keep the README's three-column table format: **Result | Paper | References and
verification**, with a PDF link in the second column. The third column may contain
a result-directory link, plain text such as "References in the PDF", or nothing.
Plain text is saved as `references_note` and displayed below the paper links;
entries without a directory link have an empty `repository` field.
Missing, empty, duplicate, or malformed tables and failed downloads
leave the existing JSON intact and fail the workflow. Intentional removal of all
entries requires editing the JSON directly.

The automation uses the repository's `GITHUB_TOKEN` to commit and dispatch the
Pages workflow. If branch protection forbids bot commits to `main`, the push will
fail and those repository rules must be adjusted before automatic updates can run.
