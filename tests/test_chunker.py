import unittest

from ingest.chunker import chunk_document
from ingest.parser import ParsedDocument


def doc(text: str, doc_id: str = "repo/path.md") -> ParsedDocument:
    return ParsedDocument(
        id=doc_id, repo="repo", path="path.md", doc_type="spec", change_id=None,
        tier=None, doc_date=None, commit_sha="0" * 40, text=text,
    )


class ChunkDocumentTest(unittest.TestCase):
    def test_one_chunk_per_h2_section_with_h1_prepended(self) -> None:
        text = "# Spec: widget\n\n## Problem\n\nThe widget is broken.\n\n## Solution\n\nFix it.\n"
        chunks = chunk_document(doc(text))
        self.assertEqual([c.heading_path for c in chunks], ["Spec: widget > Problem", "Spec: widget > Solution"])
        self.assertEqual(chunks[0].body, "The widget is broken.")
        self.assertEqual(chunks[1].body, "Fix it.")
        self.assertEqual([c.ordinal for c in chunks], [0, 1])
        self.assertEqual([c.id for c in chunks], ["repo/path.md#0", "repo/path.md#1"])

    def test_preamble_before_the_first_h2_becomes_its_own_chunk(self) -> None:
        text = "# Title\n\nIntro paragraph.\n\n## Details\n\nMore text.\n"
        chunks = chunk_document(doc(text))
        self.assertEqual([c.heading_path for c in chunks], ["Title", "Title > Details"])
        self.assertEqual(chunks[0].body, "Intro paragraph.")

    def test_a_document_with_no_headings_still_produces_at_least_one_chunk(self) -> None:
        text = "Just a plain commit message with no markdown headings at all."
        chunks = chunk_document(doc(text))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].body, text)
        self.assertEqual(chunks[0].heading_path, "Document")

    def test_an_oversized_h2_section_is_split_by_h3(self) -> None:
        big_para = "word " * 700  # well over MAX_CHUNK_WORDS
        text = f"# Doc\n\n## Big section\n\n### Part one\n\n{big_para}\n\n### Part two\n\nshort.\n"
        chunks = chunk_document(doc(text), max_words=600)
        self.assertEqual(
            [c.heading_path for c in chunks],
            ["Doc > Big section > Part one", "Doc > Big section > Part two"],
        )

    def test_a_table_that_fits_stays_whole_inside_its_chunk(self) -> None:
        table = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        text = f"# Doc\n\n## Table section\n\n{table}\n"
        chunks = chunk_document(doc(text))
        self.assertEqual(len(chunks), 1)
        self.assertIn("| 1 | 2 |", chunks[0].body)
        self.assertIn("| 3 | 4 |", chunks[0].body)

    def test_a_giant_table_is_split_by_row_with_the_header_repeated(self) -> None:
        header, sep = "| id | note |", "| --- | --- |"
        rows = [f"| {i} | {'x ' * 60} |" for i in range(30)]  # forces multiple windows
        text = "# Doc\n\n" + "\n".join([header, sep, *rows]) + "\n"
        chunks = chunk_document(doc(text), max_words=200)
        self.assertGreater(len(chunks), 1, "the table should have been split into more than one window")
        for c in chunks:
            self.assertIn(header, c.body)
            self.assertIn(sep, c.body)
        # No row appears in two different chunks, and every row appears in exactly one.
        seen = [row for c in chunks for row in c.body.splitlines() if row.startswith("| ") and row not in (header, sep)]
        self.assertEqual(sorted(seen), sorted(rows))

    def test_token_count_and_content_hash_are_stable_and_distinct_for_different_bodies(self) -> None:
        chunks = chunk_document(doc("# D\n\n## A\n\nfoo\n\n## B\n\nbar\n"))
        self.assertEqual(chunks[0].token_count, chunks[0].token_count)  # deterministic
        self.assertNotEqual(chunks[0].content_hash, chunks[1].content_hash)
        self.assertEqual(len(chunks[0].content_hash), 64)

    def test_a_heading_containing_a_gate_name_sets_the_gate_column(self) -> None:
        text = "# Doc\n\n## G4: approved\n\nnotes here\n"
        chunks = chunk_document(doc(text))
        self.assertEqual(chunks[0].gate, "G4")

    def test_a_heading_with_no_gate_leaves_gate_none(self) -> None:
        text = "# Doc\n\n## Problem\n\ntext\n"
        chunks = chunk_document(doc(text))
        self.assertIsNone(chunks[0].gate)


if __name__ == "__main__":
    unittest.main()
