#!/usr/bin/env python3
"""Validate site data and the rendered GitHub Pages artifact."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
YEAR_PATTERN = re.compile(r"^(?:19|20)\d{2}$")
LEGACY_MATHJAX = ("mathjax/2.7.1", "MathJax.Hub", "text/x-mathjax-config")


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str]] = []
        self.ids: list[str] = []
        self.link_depth = 0
        self.images_without_alt = 0
        self.linked_images_without_text = 0
        self.images_without_loading = 0
        self.headings: list[tuple[int, int]] = []
        self.article_stack: list[dict[str, int | bool]] = []
        self.articles_without_headings: list[int] = []
        self.div_aria_labels_without_role: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for attribute in ("href", "src"):
            value = values.get(attribute)
            if value:
                self.references.append((attribute, value))
        if values.get("id"):
            self.ids.append(values["id"] or "")
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append((int(tag[1]), self.getpos()[0]))
            for article in self.article_stack:
                article["has_heading"] = True
        if tag == "article":
            self.article_stack.append({"line": self.getpos()[0], "has_heading": False})
        if tag == "div" and values.get("aria-label") and not values.get("role"):
            self.div_aria_labels_without_role.append(self.getpos()[0])
        if tag == "a":
            self.link_depth += 1
        if tag == "img":
            if "alt" not in values:
                self.images_without_alt += 1
            elif self.link_depth and values.get("alt") == "":
                self.linked_images_without_text += 1
            if values.get("loading") != "lazy":
                self.images_without_loading += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.link_depth:
            self.link_depth -= 1
        if tag == "article" and self.article_stack:
            article = self.article_stack.pop()
            if not article["has_heading"]:
                self.articles_without_headings.append(int(article["line"]))


def load_array(path: Path, errors: list[str]) -> list[object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{path.relative_to(ROOT)}: {error}")
        return []
    if not isinstance(value, list):
        errors.append(f"{path.relative_to(ROOT)}: expected a JSON array")
        return []
    return value


def validate_publications(errors: list[str]) -> list[object]:
    publications = load_array(ROOT / "_data" / "publications.json", errors)
    allowed_tags = load_array(ROOT / "_data" / "publication_tags.json", errors)
    allowed_tag_set = {tag for tag in allowed_tags if isinstance(tag, str)}
    if len(allowed_tag_set) != len(allowed_tags):
        errors.append("_data/publication_tags.json: tags must be unique strings")

    for index, publication in enumerate(publications, start=1):
        location = f"_data/publications.json entry {index}"
        if not isinstance(publication, dict):
            errors.append(f"{location}: expected an object")
            continue
        for field in ("author", "title", "year"):
            if field not in publication or publication[field] in (None, ""):
                errors.append(f"{location}: missing {field!r}")
        year = str(publication.get("year", ""))
        if not YEAR_PATTERN.fullmatch(year):
            errors.append(f"{location}: invalid year {year!r}")
        tags = publication.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            errors.append(f"{location}: 'tags' must be an array of non-empty strings")
            continue
        unknown_tags = sorted(set(tags) - allowed_tag_set)
        if unknown_tags:
            errors.append(f"{location}: unknown tags: {', '.join(unknown_tags)}")
        if len(tags) != len(set(tags)):
            errors.append(f"{location}: duplicate tags")
    return publications


def validate_talks(errors: list[str]) -> None:
    talks = load_array(ROOT / "talks.json", errors)
    allowed_fields = {"title", "event", "event_url", "place", "date", "slides"}
    for index, talk in enumerate(talks, start=1):
        location = f"talks.json entry {index}"
        if not isinstance(talk, dict):
            errors.append(f"{location}: expected an object")
            continue
        if not any(talk.get(field) for field in allowed_fields):
            errors.append(f"{location}: all display fields are empty")
        for field, value in talk.items():
            if field not in allowed_fields:
                errors.append(f"{location}: unknown field {field!r}")
            elif not isinstance(value, str) or not value.strip():
                errors.append(f"{location}: {field!r} must be a non-empty string")


def local_target(site: Path, page: Path, reference: str) -> Path | None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or reference.startswith(("#", "mailto:", "tel:", "data:", "javascript:")):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    return site / path.lstrip("/") if path.startswith("/") else page.parent / path


def validate_rendered_site(site: Path, publication_count: int, errors: list[str]) -> None:
    pages = sorted(site.rglob("*.html"))
    if not pages:
        errors.append(f"{site}: no rendered HTML pages found")
        return

    for page in pages:
        source = page.read_text(encoding="utf-8", errors="replace")
        relative_page = page.relative_to(site)
        if "{%" in source or re.search(r"\{\{\s*(?:site|page|include)\b", source):
            errors.append(f"{relative_page}: unrendered Liquid markup")
        for marker in LEGACY_MATHJAX:
            if marker in source:
                errors.append(f"{relative_page}: legacy MathJax marker {marker!r}")

        parser = References()
        try:
            parser.feed(source)
            parser.close()
        except Exception as error:  # HTMLParser errors are rare but should fail CI.
            errors.append(f"{relative_page}: HTML parsing failed: {error}")
            continue

        duplicate_ids = sorted(identifier for identifier, count in Counter(parser.ids).items() if count > 1)
        if duplicate_ids:
            errors.append(f"{relative_page}: duplicate IDs: {', '.join(duplicate_ids)}")
        if parser.images_without_alt:
            errors.append(f"{relative_page}: {parser.images_without_alt} images are missing alt attributes")
        if parser.linked_images_without_text:
            errors.append(
                f"{relative_page}: {parser.linked_images_without_text} linked images have empty alt text"
            )
        previous_heading: tuple[int, int] | None = None
        for level, line in parser.headings:
            if previous_heading and level > previous_heading[0] + 1:
                errors.append(
                    f"{relative_page}:{line}: h{level} skips a level after "
                    f"h{previous_heading[0]} on line {previous_heading[1]}"
                )
            previous_heading = (level, line)
        if parser.articles_without_headings:
            lines = ", ".join(str(line) for line in parser.articles_without_headings)
            errors.append(f"{relative_page}: article elements lack headings on lines {lines}")
        if parser.div_aria_labels_without_role:
            lines = ", ".join(str(line) for line in parser.div_aria_labels_without_role)
            errors.append(f"{relative_page}: aria-label on generic div elements without roles on lines {lines}")
        if relative_page.name not in {"index.html", "about.html"} and parser.images_without_loading:
            errors.append(
                f"{relative_page}: {parser.images_without_loading} images are missing loading=\"lazy\""
            )

        for attribute, reference in parser.references:
            target = local_target(site, page, reference)
            if target is None:
                continue
            exists = target.is_file() or (target.is_dir() and (target / "index.html").is_file())
            if not exists:
                errors.append(f"{relative_page}: missing local {attribute} target {reference!r}")

    publications_page = site / "publications.html"
    home_page = site / "index.html"
    if publications_page.is_file():
        source = publications_page.read_text(encoding="utf-8")
        rendered_count = source.count("data-publication-card")
        if rendered_count != publication_count:
            errors.append(
                f"publications.html: rendered {rendered_count} publications, expected {publication_count}"
            )
    else:
        errors.append("publications.html: page was not built")
    if home_page.is_file():
        source = home_page.read_text(encoding="utf-8")
        rendered_count = source.count("data-publication-card")
        if rendered_count != min(5, publication_count):
            errors.append(f"index.html: rendered {rendered_count} recent publications, expected 5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=ROOT / "_site")
    args = parser.parse_args()

    errors: list[str] = []
    publications = validate_publications(errors)
    validate_talks(errors)
    validate_rendered_site(args.site.resolve(), len(publications), errors)

    if errors:
        print("Site validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Site validation passed: {len(publications)} publications and rendered links checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
