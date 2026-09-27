"""Deny-side tests for store/migrations/0001_init.sql's RLS policies and grants: for each of the
two application roles, one test proves what it can do, and at least one proves what it cannot --
per the security baseline's pre-ship checklist ("every new policy ships two tests: one proving it
admits the right rows, one proving it denies the wrong ones"). A blocked write is asserted as a
raised exception (GRANT-level denial here, not a silent zero-row UPDATE); a blocked read is
asserted as returning nothing, confirmed against an admin-equivalent connection so "the row
doesn't exist" and "the row is hidden from you" are never confused.

Needs STUDBOOK_INGEST_DATABASE_URL and STUDBOOK_DATABASE_URL (the two roles' own connection
strings, from store/bootstrap_db.py) in the environment. Skips the whole module, loudly, if either
is missing -- CI must set both and assert this file's test count is non-zero, never let it skip
silently green (security baseline: "CI must actually run the security tests").
"""

from __future__ import annotations

import os
import unittest
import uuid

import psycopg

from store.connect import connect

INGEST_URL = os.environ.get("STUDBOOK_INGEST_DATABASE_URL")
READER_URL = os.environ.get("STUDBOOK_DATABASE_URL")
# eval_runs is append-only to the reader role by design, so the row this suite inserts cannot be
# deleted afterwards. It carries this marker instead, so a probe row is never mistaken for a
# reported result: `select * from studbook.eval_runs where config ? 'probe'`.
PROBE_CONFIG = '{"probe": "tests/test_rls.py"}'


def probe_document(suffix: str) -> dict:
    """A throwaway, uniquely-suffixed row -- safe to insert/delete repeatedly without colliding
    with real ingested data or with a concurrent test run."""
    doc_id = f"test-rls/{suffix}-{uuid.uuid4().hex[:8]}"
    return {
        "id": doc_id,
        "repo": "test-rls",
        "path": f"{suffix}.md",
        "doc_type": "readme",
        "commit_sha": "0" * 40,
        "content_hash": "0" * 64,
    }


@unittest.skipUnless(INGEST_URL and READER_URL, "STUDBOOK_INGEST_DATABASE_URL / STUDBOOK_DATABASE_URL not set")
class RlsTest(unittest.TestCase):
    ingest_conn: psycopg.Connection
    reader_conn: psycopg.Connection

    @classmethod
    def setUpClass(cls) -> None:
        cls.ingest_conn = connect(INGEST_URL, autocommit=True)
        cls.reader_conn = connect(READER_URL, autocommit=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ingest_conn.close()
        cls.reader_conn.close()

    def insert_as_ingest(self, doc: dict) -> None:
        with self.ingest_conn.cursor() as cur:
            cur.execute(
                "insert into studbook.documents (id, repo, path, doc_type, commit_sha, content_hash) "
                "values (%(id)s, %(repo)s, %(path)s, %(doc_type)s, %(commit_sha)s, %(content_hash)s)",
                doc,
            )
        self.addCleanup(self.delete_as_ingest, doc["id"])

    def delete_as_ingest(self, doc_id: str) -> None:
        with self.ingest_conn.cursor() as cur:
            cur.execute("delete from studbook.documents where id = %s", (doc_id,))

    # ---------------------------------------------------------------- studbook_ingest: admits

    def test_ingest_role_can_insert_select_update_and_delete_its_own_row(self) -> None:
        doc = probe_document("ingest-rw")
        self.insert_as_ingest(doc)
        with self.ingest_conn.cursor() as cur:
            cur.execute("select path from studbook.documents where id = %s", (doc["id"],))
            self.assertEqual(cur.fetchone(), (doc["path"],))

            cur.execute("update studbook.documents set path = %s where id = %s", ("renamed.md", doc["id"]))
            cur.execute("select path from studbook.documents where id = %s", (doc["id"],))
            self.assertEqual(cur.fetchone(), ("renamed.md",))

            cur.execute("delete from studbook.documents where id = %s", (doc["id"],))
            cur.execute("select 1 from studbook.documents where id = %s", (doc["id"],))
            self.assertIsNone(cur.fetchone())

    # --------------------------------------------------------------- studbook_ingest: denies

    def test_ingest_role_cannot_touch_eval_runs(self) -> None:
        # No GRANT for studbook_ingest on eval_runs at all (only studbook_reader has it) -- this
        # is denied before RLS is even evaluated, so it raises rather than silently matching zero.
        with self.assertRaises(psycopg.errors.InsufficientPrivilege):
            with self.ingest_conn.cursor() as cur:
                cur.execute(
                    "insert into studbook.eval_runs (split, config_hash, config, studbook_commit, metrics) "
                    "values ('dev', %s, '{}', %s, '{}')",
                    ("0" * 64, "0" * 40),
                )

    # ---------------------------------------------------------------- studbook_reader: admits

    def test_reader_role_can_select_a_row_the_ingest_role_wrote(self) -> None:
        doc = probe_document("reader-select")
        self.insert_as_ingest(doc)
        with self.reader_conn.cursor() as cur:
            cur.execute("select path from studbook.documents where id = %s", (doc["id"],))
            self.assertEqual(cur.fetchone(), (doc["path"],))

    def test_reader_role_can_insert_and_read_back_its_own_eval_run(self) -> None:
        with self.reader_conn.cursor() as cur:
            cur.execute(
                "insert into studbook.eval_runs (split, config_hash, config, studbook_commit, metrics) "
                "values ('dev', %s, %s, %s, '{}') returning id",
                ("1" * 64, PROBE_CONFIG, "1" * 40),
            )
            (run_id,) = cur.fetchone()
            cur.execute("select split from studbook.eval_runs where id = %s", (run_id,))
            self.assertEqual(cur.fetchone(), ("dev",))
        # Deliberately not cleaned up: the reader role has no DELETE on this table and the test
        # below is what proves it. The probe row is identifiable by its `probe` config key.

    def test_reader_role_cannot_delete_or_update_an_eval_run(self) -> None:
        """The eval log is append-only by grant: a reported number must not be revisable by the
        role that reports it. A blocked DELETE would otherwise look like a no-op success, so this
        asserts the privilege error rather than a row count."""
        with self.reader_conn.cursor() as cur:
            cur.execute("select id from studbook.eval_runs order by id desc limit 1")
            row = cur.fetchone()
        if row is None:
            self.skipTest("no eval run to attempt deleting")
        for statement in ("delete from studbook.eval_runs where id = %s",
                          "update studbook.eval_runs set split = 'test' where id = %s"):
            with self.subTest(statement=statement.split()[0]):
                with self.assertRaises(psycopg.errors.InsufficientPrivilege):
                    with self.reader_conn.cursor() as cur:
                        cur.execute(statement, (row[0],))
                self.reader_conn.rollback()

    # --------------------------------------------------------------- studbook_reader: denies

    def test_reader_role_cannot_insert_a_document(self) -> None:
        doc = probe_document("reader-write-denied")
        with self.assertRaises(psycopg.errors.InsufficientPrivilege):
            with self.reader_conn.cursor() as cur:
                cur.execute(
                    "insert into studbook.documents (id, repo, path, doc_type, commit_sha, content_hash) "
                    "values (%(id)s, %(repo)s, %(path)s, %(doc_type)s, %(commit_sha)s, %(content_hash)s)",
                    doc,
                )

    def test_reader_role_cannot_update_or_delete_a_document(self) -> None:
        doc = probe_document("reader-mutate-denied")
        self.insert_as_ingest(doc)
        with self.assertRaises(psycopg.errors.InsufficientPrivilege):
            with self.reader_conn.cursor() as cur:
                cur.execute("update studbook.documents set path = 'x' where id = %s", (doc["id"],))
        with self.assertRaises(psycopg.errors.InsufficientPrivilege):
            with self.reader_conn.cursor() as cur:
                cur.execute("delete from studbook.documents where id = %s", (doc["id"],))
        # Confirmed against the ingest (write-capable) connection, not the reader's own -- a
        # blocked UPDATE/DELETE matches zero rows and reports success, so "did it change?" has to
        # be answered by someone who can actually see the row, never by re-querying as the denied
        # role (security baseline: "for a blocked UPDATE read the value back with an admin client").
        with self.ingest_conn.cursor() as cur:
            cur.execute("select path from studbook.documents where id = %s", (doc["id"],))
            self.assertEqual(cur.fetchone(), (doc["path"],), "the denied UPDATE must not have changed the row")

    # ----------------------------------------------------------------- schema-level: catalog

    def test_public_anon_and_authenticated_have_no_privileges_in_the_studbook_schema(self) -> None:
        with self.ingest_conn.cursor() as cur:
            cur.execute(
                "select grantee, table_name, privilege_type from information_schema.role_table_grants "
                "where table_schema = 'studbook' and grantee in ('PUBLIC', 'anon', 'authenticated')"
            )
            self.assertEqual(cur.fetchall(), [])

    def test_rls_is_enabled_on_every_studbook_table(self) -> None:
        with self.ingest_conn.cursor() as cur:
            cur.execute("select tablename, rowsecurity from pg_tables where schemaname = 'studbook'")
            rows = cur.fetchall()
            self.assertGreaterEqual(len(rows), 3, "expected documents, chunks and eval_runs at least")
            for tablename, rowsecurity in rows:
                self.assertTrue(rowsecurity, f"studbook.{tablename} has RLS disabled")


if __name__ == "__main__":
    unittest.main()
