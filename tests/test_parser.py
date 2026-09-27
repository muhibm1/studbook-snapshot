import subprocess
import tempfile
import unittest
from pathlib import Path

from ingest.parser import _change_id_for, _doc_type_for, walk_commits, walk_files


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


class DocTypeForTest(unittest.TestCase):
    def test_a_file_directly_in_a_change_folder(self) -> None:
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/spec.md")), "spec")
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/intent.md")), "intent")

    def test_the_two_gate_documents_since_workhorse_0_3_0(self) -> None:
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/brief.md")), "brief")
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/ship.md")), "ship")
        # The documents they replaced still ingest: older changes keep them.
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/review-packet.md")), "review-packet")

    def test_a_reviewer_report_under_a_change_folder(self) -> None:
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/reviews/wh-bug-reviewer.md")), "review")
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/reviews/spec.md")), "review")
        # Only .md files directly inside reviews/, and only inside a change folder.
        self.assertIsNone(_doc_type_for(Path("docs/sdlc/2026-01-01-x/reviews/notes.txt")))
        self.assertIsNone(_doc_type_for(Path("docs/sdlc/2026-01-01-x/reviews/deeper/x.md")))
        self.assertIsNone(_doc_type_for(Path("reviews/x.md")))

    def test_a_changelog_at_the_root_or_one_folder_down(self) -> None:
        self.assertEqual(_doc_type_for(Path("CHANGELOG.md")), "changelog")
        self.assertEqual(_doc_type_for(Path("desk/CHANGELOG.md")), "changelog")
        self.assertIsNone(_doc_type_for(Path("desk/node_modules/pkg/CHANGELOG.md")))

    def test_an_adr_under_a_change_folder(self) -> None:
        self.assertEqual(_doc_type_for(Path("docs/sdlc/2026-01-01-x/adr/0001-thing.md")), "adr")

    def test_a_design_doc(self) -> None:
        self.assertEqual(_doc_type_for(Path("docs/superpowers/specs/2026-01-01-design.md")), "design-doc")

    def test_the_repo_root_readme(self) -> None:
        self.assertEqual(_doc_type_for(Path("README.md")), "readme")

    def test_a_nested_readme_is_not_the_repo_root_readme(self) -> None:
        self.assertIsNone(_doc_type_for(Path("examples/hello-service/README.md")))

    def test_verify_logs_are_excluded(self) -> None:
        self.assertIsNone(_doc_type_for(Path("docs/sdlc/2026-01-01-x/verify-logs/test.log")))
        # Even if it somehow had a .md extension matching a known doc type's stem.
        self.assertIsNone(_doc_type_for(Path("docs/sdlc/2026-01-01-x/verify-logs/spec.md")))

    def test_codebase_map_and_constraints_at_the_sdlc_root(self) -> None:
        self.assertEqual(_doc_type_for(Path("docs/sdlc/codebase-map.md")), "codebase-map")
        self.assertEqual(_doc_type_for(Path("docs/sdlc/constraints.md")), "constraints")

    def test_an_unrecognised_file_is_skipped(self) -> None:
        self.assertIsNone(_doc_type_for(Path("docs/sdlc/2026-01-01-x/notes.md")))


class ChangeIdForTest(unittest.TestCase):
    def test_a_change_scoped_file_gets_its_folder_as_change_id(self) -> None:
        self.assertEqual(_change_id_for(Path("docs/sdlc/2026-01-01-x/spec.md"), "spec"), "2026-01-01-x")

    def test_an_adr_gets_its_change_folder_too(self) -> None:
        self.assertEqual(_change_id_for(Path("docs/sdlc/2026-01-01-x/adr/0001-y.md"), "adr"), "2026-01-01-x")

    def test_the_gate_documents_and_reviewer_reports_get_their_change_folder(self) -> None:
        self.assertEqual(_change_id_for(Path("docs/sdlc/2026-01-01-x/brief.md"), "brief"), "2026-01-01-x")
        self.assertEqual(_change_id_for(Path("docs/sdlc/2026-01-01-x/ship.md"), "ship"), "2026-01-01-x")
        self.assertEqual(_change_id_for(Path("docs/sdlc/2026-01-01-x/reviews/wh-bug-reviewer.md"), "review"),
                         "2026-01-01-x")

    def test_codebase_map_has_no_change_id_even_though_it_sits_under_sdlc(self) -> None:
        self.assertIsNone(_change_id_for(Path("docs/sdlc/codebase-map.md"), "codebase-map"))

    def test_a_design_doc_has_no_change_id(self) -> None:
        self.assertIsNone(_change_id_for(Path("docs/superpowers/specs/x.md"), "design-doc"))


class WalkGitRepoTest(unittest.TestCase):
    """Integration test against a real, disposable git repo -- walk_files/_repo_head/walk_commits
    all shell out to git, so a fake filesystem tree without a real .git can't exercise them."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "Test")

        change = self.repo / "docs" / "sdlc" / "2026-01-01-add-thing"
        change.mkdir(parents=True)
        (change / "intent.md").write_text(
            "# Intent: add thing\n\nChange id: `2026-01-01-add-thing`\nDate: 2026-01-01\n"
            "Risk tier: 2 (touches a sensitive path)\n\n## Problem\n\nNo thing exists.\n",
            encoding="utf-8",
        )
        (change / "spec.md").write_text("# Spec\n\n## Requirements\n\nR1: add the thing.\n", encoding="utf-8")
        (self.repo / "README.md").write_text("# repo\n\nA readme.\n", encoding="utf-8")

        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "feat: add the thing\n\nBecause it was missing.")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_walk_files_finds_the_change_docs_and_readme(self) -> None:
        docs = walk_files("test-repo", self.repo)
        by_path = {d.path: d for d in docs}
        self.assertEqual(set(by_path), {"docs/sdlc/2026-01-01-add-thing/intent.md",
                                         "docs/sdlc/2026-01-01-add-thing/spec.md", "README.md"})
        self.assertEqual(by_path["docs/sdlc/2026-01-01-add-thing/intent.md"].tier, 2)
        self.assertEqual(by_path["docs/sdlc/2026-01-01-add-thing/intent.md"].doc_date, "2026-01-01")

    def test_the_intent_tier_propagates_to_its_sibling_spec(self) -> None:
        docs = walk_files("test-repo", self.repo)
        spec = next(d for d in docs if d.path.endswith("spec.md"))
        self.assertEqual(spec.tier, 2)
        self.assertEqual(spec.change_id, "2026-01-01-add-thing")

    def _add_change(self, change_id: str, files: dict[str, str]) -> None:
        change = self.repo / "docs" / "sdlc" / change_id
        for rel, text in files.items():
            (change / rel).parent.mkdir(parents=True, exist_ok=True)
            (change / rel).write_text(text, encoding="utf-8")

    def test_the_brief_tier_propagates_to_every_document_of_its_change(self) -> None:
        # A WorkHorse 0.3.0 change: brief.md states the tier (templates/brief.md's second line);
        # there is no intent.md.
        self._add_change("2026-02-02-brief-only", {
            "brief.md": "# Brief\n\nChange id: `2026-02-02-brief-only` · Prepared 2026-02-02 UTC\n"
                        "Risk tier: 3 (touches production data)\n",
            "ship.md": "# Ship\n\n## Verification\n\ngreen\n",
            "reviews/wh-bug-reviewer.md": "# Bug review\n\nNo findings.\n",
        })
        by_path = {d.path: d for d in walk_files("test-repo", self.repo)}
        for rel, doc_type in (("brief.md", "brief"), ("ship.md", "ship"), ("reviews/wh-bug-reviewer.md", "review")):
            doc = by_path[f"docs/sdlc/2026-02-02-brief-only/{rel}"]
            self.assertEqual(doc.doc_type, doc_type)
            self.assertEqual(doc.change_id, "2026-02-02-brief-only")
            self.assertEqual(doc.tier, 3)

    def test_the_brief_tier_wins_over_the_intent_tier_when_both_exist(self) -> None:
        self._add_change("2026-02-03-both", {
            "intent.md": "# Intent\n\nRisk tier: 1 (old)\n",
            "brief.md": "# Brief\n\nRisk tier: 2 (redesigned)\n",
            "spec.md": "# Spec\n",
        })
        by_path = {d.path: d for d in walk_files("test-repo", self.repo)}
        self.assertEqual(by_path["docs/sdlc/2026-02-03-both/spec.md"].tier, 2)
        self.assertEqual(by_path["docs/sdlc/2026-02-03-both/intent.md"].tier, 2)

    def test_the_intent_tier_is_used_when_the_brief_states_none(self) -> None:
        self._add_change("2026-02-04-brief-without-tier", {
            "intent.md": "# Intent\n\nRisk tier: 1 (old)\n",
            "brief.md": "# Brief\n\nNo tier line yet.\n",
        })
        by_path = {d.path: d for d in walk_files("test-repo", self.repo)}
        self.assertEqual(by_path["docs/sdlc/2026-02-04-brief-without-tier/brief.md"].tier, 1)

    def test_readme_has_no_change_id_or_tier(self) -> None:
        docs = walk_files("test-repo", self.repo)
        readme = next(d for d in docs if d.path == "README.md")
        self.assertIsNone(readme.change_id)
        self.assertIsNone(readme.tier)

    def test_every_file_doc_is_stamped_with_repo_head(self) -> None:
        docs = walk_files("test-repo", self.repo)
        head = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
        self.assertTrue(all(d.commit_sha == head for d in docs))

    def test_walk_commits_returns_one_document_per_commit(self) -> None:
        commits = walk_commits("test-repo", self.repo)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].doc_type, "commit")
        self.assertIn("feat: add the thing", commits[0].text)
        self.assertIn("Because it was missing.", commits[0].text)
        self.assertIsNone(commits[0].change_id)
        self.assertIsNone(commits[0].tier)

    def test_walk_commits_handles_several_commits_with_no_cross_contamination(self) -> None:
        # Regression: git log's tformat appends a newline after every record (including after
        # this module's own trailing separator), which used to leak a leading "\n" onto every
        # entry but the first and a bogus newline-only entry at the end.
        (self.repo / "second.txt").write_text("x", encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "second commit", "-m", "second body")
        (self.repo / "third.txt").write_text("y", encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "third commit", "-m", "third body")

        commits = walk_commits("test-repo", self.repo)
        self.assertEqual(len(commits), 3)
        subjects = [c.text.splitlines()[0] for c in commits]
        self.assertEqual(subjects, ["third commit", "second commit", "feat: add the thing"])
        for c in commits:
            self.assertFalse(c.text.startswith("\n"), f"leaked leading newline in: {c.text!r}")
        shas = {c.commit_sha for c in commits}
        self.assertEqual(len(shas), 3, "each commit must get its own sha, not a shared/empty one")


if __name__ == "__main__":
    unittest.main()
