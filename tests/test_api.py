import os
import unittest
from unittest import mock

from fastapi import HTTPException

from api.main import API_KEY_ENV, MIN_API_KEY_LENGTH, configured_api_key, require_api_key
from retrieve.hybrid import EMPTY_SCOPE, Scope

READER_URL = os.environ.get("STUDBOOK_DATABASE_URL")
# A test-only key, long enough to pass the length floor, obviously not a real one.
TEST_KEY = "test-key-" + "0" * MIN_API_KEY_LENGTH


class ApiKeyTest(unittest.TestCase):
    """Deny-side first, admit-side second, per the security baseline: a policy that admits
    correctly but denies nothing looks healthy from inside the app."""

    def test_the_server_refuses_to_configure_without_a_key(self) -> None:
        # The failure the whole feature exists to prevent: an unset key must stop the server,
        # never open it. Asserted as a raise, not as "no error".
        with mock.patch.dict(os.environ, {API_KEY_ENV: ""}):
            with self.assertRaises(RuntimeError) as ctx:
                configured_api_key()
        self.assertIn(API_KEY_ENV, str(ctx.exception))
        self.assertIn("token_urlsafe", str(ctx.exception))  # tells the operator how to fix it

    def test_a_short_key_is_refused_as_if_unset(self) -> None:
        with mock.patch.dict(os.environ, {API_KEY_ENV: "hunter2"}):
            with self.assertRaises(RuntimeError):
                configured_api_key()

    def test_a_missing_header_is_401_with_a_challenge(self) -> None:
        with mock.patch.dict(os.environ, {API_KEY_ENV: TEST_KEY}):
            with self.assertRaises(HTTPException) as ctx:
                require_api_key(authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.headers.get("WWW-Authenticate"), "Bearer")

    def test_a_wrong_key_and_a_wrong_scheme_are_the_same_401(self) -> None:
        # Same status and same detail for every failure shape, so a response never says whether
        # the scheme, the header, or the key itself was the problem.
        details = set()
        with mock.patch.dict(os.environ, {API_KEY_ENV: TEST_KEY}):
            for header in (f"Bearer {TEST_KEY[:-1]}x", f"Basic {TEST_KEY}", TEST_KEY, "Bearer "):
                with self.assertRaises(HTTPException) as ctx:
                    require_api_key(authorization=header)
                self.assertEqual(ctx.exception.status_code, 401)
                details.add(ctx.exception.detail)
        self.assertEqual(len(details), 1)

    def test_the_configured_key_is_admitted(self) -> None:
        with mock.patch.dict(os.environ, {API_KEY_ENV: TEST_KEY}):
            self.assertIsNone(require_api_key(authorization=f"Bearer {TEST_KEY}"))
            self.assertIsNone(require_api_key(authorization=f"bearer {TEST_KEY}"))  # scheme is case-insensitive


class ScopeTest(unittest.TestCase):
    def test_an_empty_scope_adds_no_sql_and_no_params(self) -> None:
        self.assertEqual(EMPTY_SCOPE.where_sql(), "")
        self.assertEqual(EMPTY_SCOPE.params(), {})

    def test_a_repo_scope_adds_one_clause(self) -> None:
        scope = Scope(repo="workhorse")
        self.assertEqual(scope.where_sql(), " and d.repo = %(repo)s")
        self.assertEqual(scope.params(), {"repo": "workhorse"})

    def test_both_filters_are_anded_together(self) -> None:
        scope = Scope(repo="paddock-demo", change_id="2026-09-12-add-get-health-endpoint-returning-uptime")
        self.assertIn("d.repo = %(repo)s", scope.where_sql())
        self.assertIn("d.change_id = %(change_id)s", scope.where_sql())
        self.assertEqual(set(scope.params()), {"repo", "change_id"})

    def test_a_repos_set_restricts_to_exactly_those_repositories(self) -> None:
        # The eval's scope: the store may hold repositories the corpus never labelled (Paddock's
        # refresh adds them), and a run over the whole store is a different run.
        scope = Scope(repos=("paddock-demo", "workhorse"))
        self.assertEqual(scope.where_sql(), " and d.repo = any(%(repos)s)")
        self.assertEqual(scope.params(), {"repos": ["paddock-demo", "workhorse"]})
        # An empty tuple is a real filter that matches nothing, not "no filter".
        self.assertEqual(Scope(repos=()).params(), {"repos": []})
        self.assertEqual(Scope(repos=None).where_sql(), "")

    def test_the_clause_starts_with_and_so_it_appends_to_an_existing_where(self) -> None:
        # Both call sites splice this onto a `where` that already has a condition
        # ("where true ...", "where q.tsq is not null ...") -- a clause that didn't lead with
        # `and` would produce a syntax error only at query time, against the live database.
        self.assertTrue(Scope(repo="x").where_sql().startswith(" and "))


@unittest.skipUnless(READER_URL, "STUDBOOK_DATABASE_URL not set")
class LiveApiTest(unittest.TestCase):
    """Exercises the real app against the real store. /ask is not called here -- it costs money
    per request; tests/test_generate.py covers generation itself."""

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        from api.main import app

        # A dedicated test key, never the operator's real one: the suite must not depend on
        # whatever STUDBOOK_API_KEY happens to be set to on this machine.
        cls.env = mock.patch.dict(os.environ, {API_KEY_ENV: TEST_KEY})
        cls.env.start()
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()  # runs lifespan: checks the key, loads the models once
        cls.auth = {"Authorization": f"Bearer {TEST_KEY}"}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_ctx.__exit__(None, None, None)
        cls.env.stop()

    # ------------------------------------------------------------------ open routes

    def test_healthz_is_open_and_reports_the_corpus_and_loaded_models(self) -> None:
        body = self.client.get("/healthz").json()
        self.assertEqual(body["status"], "ok")
        self.assertGreater(body["chunks"], 0)
        self.assertGreater(body["documents"], 0)
        self.assertIn("embedder", body["models_loaded"])
        self.assertIn("reranker", body["models_loaded"])

    def test_the_index_page_is_served_without_a_key(self) -> None:
        # The shell carries nothing from the record; it is the data routes that are gated.
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Studbook", response.text)

    # ------------------------------------------------------------------ data routes: deny

    def test_every_data_route_is_401_without_a_key(self) -> None:
        for method, path, kwargs in (
            ("get", "/repos", {}),
            ("get", "/search", {"params": {"q": "gate approval"}}),
            ("post", "/ask", {"json": {"question": "why?"}}),
        ):
            with self.subTest(path=path):
                response = getattr(self.client, method)(path, **kwargs)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers.get("WWW-Authenticate"), "Bearer")
                # Denied means denied: no passage text leaks in the 401 body.
                self.assertNotIn("passages", response.text)

    def test_a_wrong_key_is_401_on_a_data_route(self) -> None:
        response = self.client.get("/repos", headers={"Authorization": f"Bearer {TEST_KEY[:-1]}x"})
        self.assertEqual(response.status_code, 401)

    def test_ask_checks_the_key_before_validating_the_body(self) -> None:
        # Order matters: an unauthenticated caller must not learn the request schema from 422s.
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 401)

    # ------------------------------------------------------------------ data routes: admit

    def test_repos_lists_both_ingested_repositories(self) -> None:
        names = {r["repo"] for r in self.client.get("/repos", headers=self.auth).json()["repos"]}
        self.assertIn("workhorse", names)
        self.assertIn("paddock-demo", names)

    def test_search_returns_numbered_passages(self) -> None:
        body = self.client.get("/search", params={"q": "package.json version endpoint", "k": 5},
                               headers=self.auth).json()
        self.assertEqual(len(body["passages"]), 5)
        self.assertEqual([p["number"] for p in body["passages"]], [1, 2, 3, 4, 5])
        self.assertTrue(all(p["body"] for p in body["passages"]))

    def test_a_repo_scope_filter_actually_restricts_results(self) -> None:
        body = self.client.get("/search", params={"q": "gate approval", "repo": "workhorse", "k": 5},
                               headers=self.auth).json()
        self.assertTrue(body["passages"], "expected some workhorse passages")
        self.assertEqual({p["repo"] for p in body["passages"]}, {"workhorse"})

    def test_a_scope_filter_for_a_repo_with_no_match_returns_nothing(self) -> None:
        body = self.client.get("/search", params={"q": "gate approval", "repo": "no-such-repo"},
                               headers=self.auth).json()
        self.assertEqual(body["passages"], [])

    def test_ask_rejects_an_empty_question_without_calling_the_model(self) -> None:
        self.assertEqual(self.client.post("/ask", json={"question": ""}, headers=self.auth).status_code, 422)


if __name__ == "__main__":
    unittest.main()


class SecurityHeadersTest(unittest.TestCase):
    """Every response carries the headers, and the page needs nothing the CSP forbids. The second
    half matters as much as the first: a strict policy on a page with an inline script is a page
    that silently does nothing."""

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from api.main import app
        self.client = TestClient(app)  # no `with`: lifespan (model loading) is not needed here

    def test_the_page_and_the_api_both_carry_the_headers(self) -> None:
        from api.main import SECURITY_HEADERS
        with mock.patch.dict(os.environ, {API_KEY_ENV: TEST_KEY}):
            for path in ("/", "/static/app.js", "/repos"):
                with self.subTest(path):
                    response = self.client.get(path)  # /repos is a 401 without a key: still covered
                    for name, value in SECURITY_HEADERS.items():
                        self.assertEqual(response.headers.get(name), value)

    def test_the_policy_admits_no_inline_code(self) -> None:
        from api.main import SECURITY_HEADERS
        csp = SECURITY_HEADERS["Content-Security-Policy"]
        self.assertNotIn("unsafe-inline", csp)
        self.assertNotIn("unsafe-eval", csp)
        self.assertIn("frame-ancestors 'none'", csp)

    def test_the_page_itself_has_no_inline_script_style_or_handlers(self) -> None:
        import re
        from pathlib import Path
        html = (Path(__file__).resolve().parent.parent / "api" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotRegex(html, r"<script(?![^>]*\bsrc=)[^>]*>")
        self.assertNotIn("<style", html)
        self.assertIsNone(re.search(r"\son[a-z]+\s*=", html), "inline event handler")
        self.assertNotIn('style="', html)
