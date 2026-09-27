"""The workflow files parse, and keep the promises docs/ci.md makes about them.

The first push of ci.yml failed in 0 seconds on an unquoted colon: GitHub rejects a workflow that
does not parse without running anything, and says only "workflow file issue". This catches that
locally, along with the two properties that make the CI worth trusting."""

import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


class WorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))

    def test_every_workflow_parses(self) -> None:
        for path in WORKFLOWS.glob("*.yml"):
            with self.subTest(path.name):
                self.assertIsInstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)

    def test_the_live_job_never_reads_production_secrets(self) -> None:
        env = self.ci["jobs"]["live"]["env"]
        for name, value in env.items():
            with self.subTest(name):
                self.assertIn("secrets.STUDBOOK_TEST_", value)

    def test_every_action_is_pinned_to_a_full_commit_sha(self) -> None:
        # A tag can be moved to different code after review; a commit sha cannot.
        import re
        for job in self.ci["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    with self.subTest(step["uses"]):
                        self.assertRegex(step["uses"], re.compile(r"@[0-9a-f]{40}$"))

    def test_the_checks_job_has_no_credentials(self) -> None:
        text = yaml.safe_dump(self.ci["jobs"]["checks"])
        self.assertNotIn("secrets.", text)


if __name__ == "__main__":
    unittest.main()
