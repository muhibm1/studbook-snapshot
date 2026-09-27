"""store/connect.py: every connection verifies the server against the pinned Supabase root.

Deny-side first, per the security baseline. A verified connection that "works" proves little on
its own -- a connection that never verified would work identically. What proves verification is
on is that a wrong root makes the same connection FAIL."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import psycopg

from store import connect as connect_module
from store.connect import CA_CERT, SSL_MODE, connect

READER_URL = os.environ.get("STUDBOOK_DATABASE_URL")


class ConnectArgumentsTest(unittest.TestCase):
    def test_it_forces_verify_full_and_the_pinned_root(self) -> None:
        captured = {}

        def fake_connect(conninfo, **kwargs):
            captured.update(kwargs, conninfo=conninfo)
            return "conn"

        with mock.patch.object(psycopg, "connect", fake_connect):
            self.assertEqual(connect("host=x user=y", autocommit=True), "conn")
        self.assertEqual(captured["sslmode"], "verify-full")
        self.assertEqual(captured["sslrootcert"], str(CA_CERT))
        self.assertTrue(captured["autocommit"])  # caller options still pass through
        self.assertEqual(captured["conninfo"], "host=x user=y")

    def test_the_pinned_root_is_committed_and_is_a_certificate(self) -> None:
        self.assertTrue(CA_CERT.is_file(), f"{CA_CERT} missing")
        text = CA_CERT.read_text(encoding="ascii")
        self.assertTrue(text.startswith("-----BEGIN CERTIFICATE-----"))
        self.assertEqual(SSL_MODE, "verify-full")

    def test_a_missing_root_refuses_rather_than_connecting_unverified(self) -> None:
        # The one way this helper could regress to the old behaviour is by quietly dropping the
        # root; that must be a hard stop, not a downgrade.
        with mock.patch.object(connect_module, "CA_CERT", Path(tempfile.gettempdir()) / "no-such-ca.pem"):
            with mock.patch.object(psycopg, "connect") as never:
                with self.assertRaises(RuntimeError):
                    connect("host=x")
                never.assert_not_called()


@unittest.skipUnless(READER_URL, "STUDBOOK_DATABASE_URL not set")
class LiveVerificationTest(unittest.TestCase):
    def test_a_wrong_root_is_rejected(self) -> None:
        # Same host, same credentials, a genuine self-signed root that signed nothing Supabase
        # presents (tests/fixtures/unrelated-root.pem, generated with openssl for a throwaway CN).
        # The handshake must fail. This is the test that distinguishes "verifying" from
        # "encrypting": a fabricated or unparseable PEM would also fail, but for the wrong reason,
        # which is why the fixture is a real certificate.
        wrong = Path(__file__).with_name("fixtures") / "unrelated-root.pem"
        self.assertIn("BEGIN CERTIFICATE", wrong.read_text(encoding="ascii"))
        with self.assertRaises(psycopg.OperationalError) as ctx:
            psycopg.connect(READER_URL, sslmode="verify-full", sslrootcert=str(wrong), connect_timeout=15)
        self.assertIn("certificate", str(ctx.exception).lower())

    def test_the_pinned_root_is_accepted_and_the_session_is_tls(self) -> None:
        with connect(READER_URL, autocommit=True, connect_timeout=15) as conn:
            self.assertEqual(conn.execute("select 1").fetchone(), (1,))
            # Checked on the client, via libpq: this is THIS connection's TLS state. pg_stat_ssl
            # would be the wrong instrument through Supavisor -- the pooler terminates TLS, so the
            # backend's row describes the pooler's internal hop and reads false even when the
            # client hop is verified (observed 2026-09-17, first version of this test).
            self.assertTrue(conn.info.pgconn.ssl_in_use)


if __name__ == "__main__":
    unittest.main()


class ReaderSessionTest(unittest.TestCase):
    def test_the_session_sets_the_schema_and_pgvectors_iterative_scan(self) -> None:
        # Twenty hand-written copies of the first half existed before this; the second half was in
        # none of them (store/connect.py READER_SESSION).
        from store.connect import READER_SESSION, prepare_session

        executed = []

        class Cursor:
            def execute(self, sql):
                executed.append(sql)

        prepare_session(Cursor())
        self.assertEqual(executed, [READER_SESSION])
        self.assertIn("set search_path to studbook, extensions", READER_SESSION)
        self.assertIn("set hnsw.iterative_scan = strict_order", READER_SESSION)
