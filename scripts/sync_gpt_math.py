#!/usr/bin/env python3
"""Sync the GPT_Math README results table into the site's JSON data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/beniamin-bogosel/GPT_Math"
BRANCH = "master"
README_URL = f"https://raw.githubusercontent.com/beniamin-bogosel/GPT_Math/{BRANCH}/README.md"
LINK = re.compile(r"\[([^\]]+)\]\(<?([^\s)>]+)>?\)")


def cells(line: str) -> list[str]:
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def result_link(cell: str, *, pdf: bool = False) -> str:
    match = LINK.search(cell)
    if not match:
        raise ValueError(f"Missing Markdown link in table cell: {cell!r}")
    base = f"{REPOSITORY}/{'raw/refs/heads' if pdf else 'tree'}/{BRANCH}/"
    url = urljoin(base, match[2])
    if pdf:
        url = url.replace(f"{REPOSITORY}/blob/{BRANCH}/", base, 1)
    if urlsplit(url).scheme not in {"https", "http"}:
        raise ValueError(f"Unsupported link: {url!r}")
    if pdf and not urlsplit(url).path.lower().endswith(".pdf"):
        raise ValueError(f"Paper link is not a PDF: {url!r}")
    return url


def parse_results(markdown: str) -> list[dict[str, str]]:
    lines = markdown.splitlines()
    for index, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        header = [cell.lower() for cell in cells(line)]
        if header[:2] != ["result", "paper"] or len(header) != 3:
            continue
        separator = cells(lines[index + 1])
        if len(separator) != 3 or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
            continue
        results = []
        for row in lines[index + 2:]:
            if not row.strip() or "|" not in row:
                break
            values = cells(row)
            if len(values) != 3 or not values[0]:
                raise ValueError(f"Malformed results-table row: {row!r}")
            result = {
                "title": LINK.sub(r"\1", values[0]).strip(),
                "pdf": result_link(values[1], pdf=True),
                "repository": result_link(values[2]) if LINK.search(values[2]) else "",
            }
            if values[2] and not result["repository"]:
                result["references_note"] = values[2]
            results.append(result)
        if not results:
            raise ValueError("Results table is empty; refusing to clear existing entries.")
        for field in ("pdf", "repository"):
            links = [result[field] for result in results if result[field]]
            if len(set(links)) != len(links):
                raise ValueError(f"Duplicate {field} links in results table.")
        return results
    raise ValueError("Could not find the Result / Paper / References results table.")


def merge_results(results: list[dict[str, str]], existing: list[dict]) -> list[dict]:
    """Refresh table fields, preserve local metadata, and follow table order/removals."""
    merged = []
    for result in results:
        matches = [entry for entry in existing if (
            (result["repository"] and entry.get("repository") == result["repository"])
            or entry.get("pdf") == result["pdf"]
        )]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous existing entries for {result['title']!r}.")
        entry = dict(matches[0]) if matches else {
            "author": "Beniamin Bogosel",
            "year": str(datetime.now(timezone.utc).year),
        }
        entry.pop("references_note", None)
        entry.update(result)
        merged.append(entry)
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, help="Read a local README instead of fetching GitHub")
    parser.add_argument("--output", type=Path, default=ROOT / "_data" / "gpt_math.json")
    parser.add_argument("--check", action="store_true", help="Report pending changes without writing (exit 1 if out of date)")
    args = parser.parse_args(argv)

    try:
        if args.readme:
            markdown = args.readme.read_text(encoding="utf-8")
        else:
            request = Request(README_URL, headers={"User-Agent": "GPT-Math-site-sync"})
            with urlopen(request, timeout=30) as response:
                markdown = response.read().decode("utf-8")
        results = parse_results(markdown)
        existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else []
        if not isinstance(existing, list) or any(not isinstance(entry, dict) for entry in existing):
            raise ValueError("Existing JSON must contain an array of result objects.")
        merged = merge_results(results, existing)
        if merged == existing:
            print(f"GPT Math is up to date ({len(merged)} entries).")
            return 0
        if args.check:
            print(f"GPT Math needs updating ({len(merged)} entries in the repository table).")
            return 1
        # Fetching, parsing, and merging must all succeed before touching the JSON.
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(f"Updated {args.output} ({len(merged)} entries).")
        return 0
    except (OSError, ValueError) as error:
        print(f"GPT Math sync failed; existing JSON was kept: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
