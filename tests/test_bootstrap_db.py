"""Regression test for the incident on 2026-09-15: connection_url() used to do its own string
surgery with urllib.parse.urlsplit/urlunsplit, which is a generic-URI parser, not a libpq one. It
choked on a password containing characters libpq's URI form requires percent-encoded, and its
ValueError carried a fragment of the password into the traceback -- which reached stdout, in a
Supabase-generated admin database password, in a live session. These tests exist so that class of
bug cannot silently come back: every case here is a password shape that must never raise, and the
round trip must return exactly the password given, never a mangled or truncated one."""

import unittest
from pathlib import Path

from psycopg.conninfo import conninfo_to_dict

from store.bootstrap_db import Route, admin_url_for_route, connection_url, project_ref_from_host

ADMIN_URL = "postgresql://postgres:oldpassword@db.example.supabase.co:5432/postgres"
DIRECT_ROUTE = Route(host="db.example.supabase.co", port="5432", project_ref=None)
POOLER_ROUTE = Route(host="aws-0-us-east-1.pooler.supabase.com", port="5432", project_ref="example")

# Every one of these needs percent-encoding to be a *valid* libpq URI if hand-built by string
# concatenation; conninfo_to_dict/make_conninfo must handle them with no encoding step of our own.
TRICKY_PASSWORDS = [
    "plainAlnum123",
    "has@an-at-sign",
    "has:a:colon",
    "has/a/slash",
    "has%25a-percent-sign",
    "has[brackets]in-it",
    "has spaces in it",
    "has'a\"quote",
    "endsWithPercent%",
]


class ConnectionUrlTest(unittest.TestCase):
    def test_every_tricky_password_round_trips_without_raising(self) -> None:
        for password in TRICKY_PASSWORDS:
            with self.subTest(password=password):
                url = connection_url(ADMIN_URL, DIRECT_ROUTE, "studbook_reader_login", password)
                fields = conninfo_to_dict(url)
                self.assertEqual(fields["password"], password)
                self.assertEqual(fields["user"], "studbook_reader_login")

    def test_host_port_and_dbname_are_preserved_from_the_admin_url(self) -> None:
        url = connection_url(ADMIN_URL, DIRECT_ROUTE, "studbook_ingest_login", "x")
        fields = conninfo_to_dict(url)
        self.assertEqual(fields["host"], "db.example.supabase.co")
        self.assertEqual(fields["port"], "5432")
        self.assertEqual(fields["dbname"], "postgres")

    def test_a_tricky_password_never_appears_unencoded_in_the_serialised_url(self) -> None:
        # The specific failure mode: an unencoded special character corrupting the string so
        # badly that a *different* substring (not even the real password) leaks into an error.
        # This asserts the round trip is exact, not merely "did not crash".
        password = "@:/%[] weird"
        url = connection_url(ADMIN_URL, DIRECT_ROUTE, "studbook_reader_login", password)
        self.assertEqual(conninfo_to_dict(url)["password"], password)

    def test_a_pooler_route_suffixes_the_username_with_the_project_ref(self) -> None:
        url = connection_url(ADMIN_URL, POOLER_ROUTE, "studbook_reader_login", "x")
        fields = conninfo_to_dict(url)
        self.assertEqual(fields["user"], "studbook_reader_login.example")
        self.assertEqual(fields["host"], "aws-0-us-east-1.pooler.supabase.com")

    def test_a_direct_route_does_not_suffix_the_username(self) -> None:
        url = connection_url(ADMIN_URL, DIRECT_ROUTE, "studbook_reader_login", "x")
        self.assertEqual(conninfo_to_dict(url)["user"], "studbook_reader_login")


class AdminUrlForRouteTest(unittest.TestCase):
    def test_a_pooler_route_suffixes_the_admin_username_too(self) -> None:
        # Regression: run() used to rehome host/port onto the route but leave the admin user as
        # bare "postgres", which connects (the pooler doesn't reject the TCP handshake) and then
        # fails mid-session with Supavisor's "no tenant identifier provided" -- a failure mode
        # `probe_route` never sees, because it builds its own trial URL correctly.
        url = admin_url_for_route(ADMIN_URL, POOLER_ROUTE)
        fields = conninfo_to_dict(url)
        self.assertEqual(fields["user"], "postgres.example")
        self.assertEqual(fields["host"], "aws-0-us-east-1.pooler.supabase.com")
        self.assertEqual(fields["password"], "oldpassword")

    def test_a_direct_route_leaves_the_admin_username_bare(self) -> None:
        url = admin_url_for_route(ADMIN_URL, DIRECT_ROUTE)
        self.assertEqual(conninfo_to_dict(url)["user"], "postgres")


class ProjectRefFromHostTest(unittest.TestCase):
    def test_extracts_the_ref_from_a_direct_connection_host(self) -> None:
        self.assertEqual(project_ref_from_host("db.productionprojectref.supabase.co"), "productionprojectref")

    def test_rejects_a_host_that_is_not_the_direct_connection_shape(self) -> None:
        with self.assertRaises(ValueError):
            project_ref_from_host("aws-0-us-east-1.pooler.supabase.com")


if __name__ == "__main__":
    unittest.main()


class PortabilityTest(unittest.TestCase):
    """The module imports anywhere; only storing a password is Windows-specific, and elsewhere it
    refuses loudly rather than leaking the value. The first Linux CI run could not import it."""

    def test_storing_a_password_off_windows_refuses_without_printing_it(self) -> None:
        from unittest import mock

        import store.bootstrap_db as b
        with mock.patch.object(b.sys, "platform", "linux"):
            with self.assertRaises(RuntimeError) as caught:
                b.set_user_env_var("STUDBOOK_TEST_ROLE", "hunter2-secret")
        self.assertIn("not Windows", str(caught.exception))
        self.assertNotIn("hunter2-secret", str(caught.exception))


class TargetTest(unittest.TestCase):
    """Pointed at a test project, this script once would have overwritten production's
    connection strings: one hard-coded table of env var names. These hold the separation."""

    def test_the_test_target_writes_no_production_env_var(self) -> None:
        from store.bootstrap_db import TARGETS
        prod_admin, prod_roles = TARGETS["production"]
        test_admin, test_roles = TARGETS["test"]
        prod_names = {prod_admin, *(e for _, _, e in prod_roles)}
        test_names = {test_admin, *(e for _, _, e in test_roles)}
        self.assertFalse(prod_names & test_names)
        self.assertTrue(all(n.startswith("STUDBOOK_TEST_") for n in test_names))

    def test_the_test_names_are_the_ones_ci_reads(self) -> None:
        # .github/workflows/ci.yml's live job reads exactly these (without the _ADMIN one).
        from store.bootstrap_db import TARGETS
        self.assertEqual({e for _, _, e in TARGETS["test"][1]},
                         {"STUDBOOK_TEST_INGEST_DATABASE_URL", "STUDBOOK_TEST_DATABASE_URL"})

    def test_a_test_run_against_the_production_project_is_refused(self) -> None:
        from store.bootstrap_db import refuse_production
        prod_reader = "host=aws-0-us-east-1.pooler.supabase.com user=studbook_reader_login.prodref password=x dbname=postgres"
        with self.assertRaises(SystemExit):
            refuse_production("test", "host=db.prodref.supabase.co user=postgres password=y dbname=postgres", prod_reader)
        refuse_production("test", "host=db.testref.supabase.co user=postgres password=y dbname=postgres", prod_reader)
        refuse_production("production", "host=db.prodref.supabase.co user=postgres password=y dbname=postgres", prod_reader)


class MigrationsOnlyTest(unittest.TestCase):
    """--migrations-only must never rotate a role: rotating changes the reader's connection string,
    and Paddock keeps its own encrypted copy of it, which would silently stop working."""

    def _main(self, argv: list[str]) -> list:
        from unittest import mock
        import store.bootstrap_db as b
        calls = []
        env = {"STUDBOOK_ADMIN_DATABASE_URL": "host=db.prodref.supabase.co user=postgres password=x dbname=postgres",
               "STUDBOOK_TEST_ADMIN_DATABASE_URL": "host=db.testref.supabase.co user=postgres password=x dbname=postgres"}
        with mock.patch.object(b.sys, "argv", ["bootstrap_db", *argv]), \
             mock.patch.dict(b.os.environ, env), \
             mock.patch.object(b, "run", lambda *a: calls.append(a)):
            self.assertEqual(b.main(), 0)
        return calls[0]

    def test_migrations_only_does_not_rotate_roles(self) -> None:
        self.assertIs(self._main(["--migrations-only"])[2], False)

    def test_a_plain_run_rotates_roles(self) -> None:
        self.assertIs(self._main([])[2], True)

    def test_flags_combine_in_either_order(self) -> None:
        from store.bootstrap_db import TARGETS
        for argv in (["--target", "test", "--migrations-only"], ["--migrations-only", "--target", "test"]):
            with self.subTest(argv):
                admin_url, roles, rotate = self._main(argv)
                self.assertIn("testref", admin_url)
                self.assertEqual(roles, TARGETS["test"][1])
                self.assertIs(rotate, False)


class BackfillTest(unittest.TestCase):
    """The one-time ledger backfill is for 0001 only. It once matched every unrecorded migration,
    so 0002 was recorded as applied on both projects without running."""

    class Cursor:
        def __init__(self) -> None:
            self.sql: list[str] = []

        def execute(self, sql, params=None) -> None:
            self.sql.append(sql)

        def fetchone(self):
            return (True,)  # studbook.documents exists, as it always does after 0001

    def test_a_later_migration_is_never_backfilled(self) -> None:
        from store.bootstrap_db import backfill_if_already_applied
        cur = self.Cursor()
        self.assertFalse(backfill_if_already_applied(cur, str(Path("store/migrations/0002_changelog_doc_type.sql")), "h"))
        self.assertEqual(cur.sql, [], "must not even look, let alone write the ledger")

    def test_0001_is_still_recoverable(self) -> None:
        from store.bootstrap_db import backfill_if_already_applied
        self.assertTrue(backfill_if_already_applied(self.Cursor(), str(Path("store/migrations/0001_init.sql")), "h"))
