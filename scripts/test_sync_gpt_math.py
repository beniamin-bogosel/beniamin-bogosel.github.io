"""Checks for table changes, preserved metadata, and failed syncs."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from sync_gpt_math import main, merge_results, parse_results


TABLE = """# Mathematical results

| Result | Paper | References and verification |
| --- | --- | --- |
| First note | [PDF](Results/First/paper.pdf) | [Result directory](Results/First) |
"""
SECOND = "| Second note | [PDF](Results/Second/paper.pdf) | [Result directory](Results/Second) |\n"


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(StringIO()))
        self.enterContext(redirect_stderr(StringIO()))

    def test_table_changes_preserve_metadata_and_remove_missing_entries(self):
        result = parse_results(TABLE)[0]
        existing = [dict(result, author="A. Author", year="2024", abstract="Edited abstract", revised="2024-01-02", ai_assistance="AI credit")]
        existing.append({"title": "Removed", "pdf": "https://example.org/removed.pdf"})
        changed = TABLE.replace("First note", "Revised title").replace("First/paper.pdf", "First/revised.pdf") + SECOND
        merged = merge_results(parse_results(changed), existing)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["title"], "Revised title")
        self.assertTrue(merged[0]["pdf"].endswith("/First/revised.pdf"))
        for key in ("author", "year", "abstract", "revised", "ai_assistance"):
            self.assertEqual(merged[0][key], existing[0][key])
        self.assertEqual(merged[1]["author"], "Beniamin Bogosel")
        self.assertRegex(merged[1]["year"], r"^20\d{2}$")
        self.assertNotIn("abstract", merged[1])

    def test_reordering_and_directory_changes_keep_metadata(self):
        results = parse_results(TABLE + SECOND)
        existing = [dict(result, abstract=result["title"]) for result in results]
        results.reverse()
        results[1]["repository"] += "Moved"
        merged = merge_results(results, existing)
        self.assertEqual([entry["abstract"] for entry in merged], ["Second note", "First note"])

    def test_absolute_pdf_links_and_escaped_pipes(self):
        table = TABLE.replace("First note", r"First \| note").replace(
            "Results/First/paper.pdf", "https://github.com/beniamin-bogosel/GPT_Math/blob/master/Results/First/paper.pdf")
        result = parse_results(table)[0]
        self.assertEqual(result["title"], "First | note")
        self.assertIn("/raw/refs/heads/master/", result["pdf"])

    def test_outer_table_pipes_are_optional(self):
        table = "\n".join(line.strip().strip("|") for line in TABLE.splitlines())
        self.assertEqual(parse_results(table), parse_results(TABLE))

    def test_multiple_text_only_reference_cells_sync_without_ambiguous_matches(self):
        note = "References and verification details in the PDF"
        table = (TABLE + SECOND).replace("[Result directory](Results/First)", note).replace(
            "[Result directory](Results/Second)", "References and complete proof in the PDF")
        results = parse_results(table)
        self.assertEqual([result["repository"] for result in results], ["", ""])
        self.assertEqual(results[0]["references_note"], note)
        existing = [dict(result, abstract=result["title"]) for result in results]
        self.assertEqual(merge_results(results, existing), existing)

    def test_reference_links_can_be_replaced_with_notes_and_restored(self):
        result = parse_results(TABLE)[0]
        existing = [dict(result, abstract="Keep this abstract")]
        table = TABLE.replace("[Result directory](Results/First)", "Details in the PDF")
        merged = merge_results(parse_results(table), existing)
        self.assertEqual(merged[0]["repository"], "")
        self.assertEqual(merged[0]["references_note"], "Details in the PDF")
        self.assertEqual(merged[0]["abstract"], "Keep this abstract")
        restored = merge_results(parse_results(TABLE), merged)
        self.assertEqual(restored, existing)

    def test_empty_reference_cell_is_allowed(self):
        result = parse_results(TABLE.replace("[Result directory](Results/First)", ""))[0]
        self.assertEqual(result["repository"], "")
        self.assertNotIn("references_note", result)

    def test_malformed_empty_and_duplicate_tables_are_rejected(self):
        for table in ("No table", TABLE.rsplit("| First note", 1)[0], TABLE + TABLE.splitlines()[-1], TABLE + SECOND.replace("[PDF]", "PDF")):
            with self.subTest(table=table), self.assertRaises(ValueError):
                parse_results(table)

    def test_failed_fetch_or_bad_table_never_overwrites_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            output.write_text('[{"title": "Keep me"}]\n')
            before = output.read_bytes()
            with patch("sync_gpt_math.urlopen", side_effect=URLError("offline")):
                self.assertEqual(main(["--output", str(output)]), 1)
            self.assertEqual(output.read_bytes(), before)
            readme = Path(directory) / "README.md"
            readme.write_text(TABLE + SECOND.replace("[PDF]", "PDF"))
            self.assertEqual(main(["--readme", str(readme), "--output", str(output)]), 1)
            self.assertEqual(output.read_bytes(), before)

    def test_check_mode_and_repeated_sync_do_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            output.write_text("[]\n")
            readme = Path(directory) / "README.md"
            readme.write_text(TABLE)
            arguments = ["--readme", str(readme), "--output", str(output)]
            self.assertEqual(main(arguments + ["--check"]), 1)
            self.assertEqual(json.loads(output.read_text()), [])
            self.assertEqual(main(arguments), 0)
            before = output.stat().st_mtime_ns
            self.assertEqual(main(arguments), 0)
            self.assertEqual(main(arguments + ["--check"]), 0)
            self.assertEqual(output.stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
