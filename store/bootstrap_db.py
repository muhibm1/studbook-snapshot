"""Apply store/migrations/*.sql to the studbook Supabase project as the admin role, then provision
(or rotate) the two login roles the application actually connects as, and store their connection
strings as Windows User environment variables -- never in a file, never printed.

    .venv/Scripts/python.exe -m store.bootstrap_db                  # production
    .venv/Scripts/python.exe -m store.bootstrap_db --target test    # CI's test project: STUDBOOK_TEST_*
    ... --migrations-only     # apply schema changes, leave every role and connection string alone

Requires STUDBOOK_ADMIN_DATABASE_URL (the project's `postgres` connection string, direct-connection
form: db.<ref>.supabase.co) in the calling process's environment. Because Windows only loads User
environment variables into a process at process start, and both `STUDBOOK_INGEST_DATABASE_URL` and
`STUDBOOK_DATABASE_URL` are written by *this* run, the session that invoked this script needs
restarting before those two are visible to it -- this script prints that reminder, never the
values.

Network note: the direct-connection host (`db.<ref>.supabase.co`) resolves to an IPv6-only
address. On a network with no IPv6 route (confirmed the case on the machine this was written on),
that connection fails outright, before authentication -- so this script also tries Supabase's
Supavisor pooler (`aws-0-<region>.pooler.supabase.com` / `aws-1-...`, IPv4), which requires the
username suffixed `.{project_ref}`. It tries direct first, then each pooler host, and uses whichever
one actually connects for every connection string it produces (the app should get a connection
string it can actually route to, not the one the schema owner happened to use).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from store.connect import connect

ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = sorted((ROOT / "store" / "migrations").glob("*.sql"))

# Per target: the env var the admin connection string is read from, and for each login role
# (login role, group role it inherits, the env var its connection string is stored under).
#
# The test target exists because CI's live job needs a separate project (docs/ci.md), and until
# 2026-09-19 this script had one hard-coded table: pointed at a test project, it would have
# overwritten the PRODUCTION connection strings with the test project's. The two tables share no
# env var name (tests/test_bootstrap_db.py holds that), and a test run refuses outright if its admin
# connection names the same Supabase project as the production reader (`refuse_production`).
TARGETS = {
    "production": ("STUDBOOK_ADMIN_DATABASE_URL", [
        ("studbook_ingest_login", "studbook_ingest", "STUDBOOK_INGEST_DATABASE_URL"),
        ("studbook_reader_login", "studbook_reader", "STUDBOOK_DATABASE_URL"),
    ]),
    "test": ("STUDBOOK_TEST_ADMIN_DATABASE_URL", [
        ("studbook_ingest_login", "studbook_ingest", "STUDBOOK_TEST_INGEST_DATABASE_URL"),
        ("studbook_reader_login", "studbook_reader", "STUDBOOK_TEST_DATABASE_URL"),
    ]),
}
ROLES = TARGETS["production"][1]

POOLER_REGION = "us-east-1"  # matches the project's region (confirmed via the Supabase API)
POOLER_HOSTS = [f"aws-{i}-{POOLER_REGION}.pooler.supabase.com" for i in (0, 1)]
CONNECT_TIMEOUT_S = 8


@dataclass(frozen=True)
class Route:
    """How to reach the database: which host/port, and whether Supavisor's `user.{project_ref}`
    suffix is needed (only when going through the pooler, never on the direct host)."""

    host: str
    port: str
    project_ref: str | None  # set only for a pooler route

    def username(self, role: str) -> str:
        return f"{role}.{self.project_ref}" if self.project_ref else role


def project_ref_from_host(host: str) -> str:
    """`db.<ref>.supabase.co` -> `<ref>`. The direct-connection host format is fixed by Supabase;
    this does not need to handle arbitrary hosts."""
    parts = host.split(".")
    if len(parts) < 2 or parts[0] != "db":
        raise ValueError(f"expected a direct-connection host like db.<ref>.supabase.co, got {host!r}")
    return parts[1]


def project_ref_of(url: str) -> str | None:
    """The Supabase project a connection string points at: from a direct host (db.<ref>...), or
    from a pooler username (<role>.<ref>). None when neither says."""
    fields = conninfo_to_dict(url)
    host, user = fields.get("host", ""), fields.get("user", "")
    if host.startswith("db."):
        return project_ref_from_host(host)
    if "." in user:
        return user.split(".", 1)[1]
    return None


def refuse_production(target: str, admin_url: str, production_reader_url: str | None) -> None:
    """A test run must not touch the production project, whatever its env vars say."""
    if target != "test" or not production_reader_url:
        return
    test_ref, prod_ref = project_ref_of(admin_url), project_ref_of(production_reader_url)
    if test_ref and test_ref == prod_ref:
        raise SystemExit("refusing: STUDBOOK_TEST_ADMIN_DATABASE_URL points at the production project "
                         "(the same project as STUDBOOK_DATABASE_URL). A test target needs its own project.")


def probe_route(admin_url: str) -> Route:
    """Find a route this machine can actually connect through, trying the direct host first (the
    normal case) and falling back to the Supavisor pooler. Never logs the admin URL or password;
    reports only which host each attempt used and whether it connected."""
    fields = conninfo_to_dict(admin_url)
    direct_host, port = fields["host"], fields.get("port", "5432")
    project_ref = project_ref_from_host(direct_host)

    candidates: list[Route] = [Route(direct_host, port, project_ref=None)]
    candidates += [Route(host, "5432", project_ref=project_ref) for host in POOLER_HOSTS]

    for route in candidates:
        trial_fields = dict(fields, host=route.host, port=route.port, user=route.username(fields["user"]))
        try:
            with connect(make_conninfo(**trial_fields), connect_timeout=CONNECT_TIMEOUT_S):
                print(f"route: connected via {route.host}:{route.port}")
                return route
        except psycopg.OperationalError as e:
            # Safe to print in full: libpq's own errors (auth failures, timeouts, DNS, refused
            # connections) never echo the password back -- that class of leak was specifically in
            # the generic string parser this module no longer uses (see connection_url's
            # docstring). This message is exactly what distinguishes "wrong password" from
            # "can't reach this host" from "wrong username for this route", which the caller needs.
            print(f"route: {route.host}:{route.port} did not connect: {e}")
    raise SystemExit("no route (direct or pooler) could connect -- check network and credentials")


def ensure_migrations_table(cur: psycopg.Cursor) -> None:
    """A ledger of which migration files have been applied, so re-running this script is safe --
    matches the standard schema-migration idiom (Flyway/Alembic/...); the versioned migrations
    themselves don't need to be individually idempotent."""
    cur.execute("create schema if not exists studbook_meta")
    cur.execute(
        "create table if not exists studbook_meta.migrations "
        "(filename text primary key, content_hash text not null, applied_at timestamptz not null default now())"
    )


def backfill_if_already_applied(cur: psycopg.Cursor, filename: str, content_hash: str) -> bool:
    """One-time recovery for a single incident (2026-09-15): this migration's DDL was applied and
    committed before the migrations table above existed (a later step in that same run failed and
    aborted before recording it). Detects that specific case -- the migration's own marker table
    already present -- and backfills the ledger instead of trying to re-run DDL that would now
    fail with "already exists". Returns whether it backfilled (i.e., the caller should skip
    re-running this file's DDL).

    Only ever for 0001_init.sql. Until 2026-09-19 it matched ANY unrecorded migration whenever
    studbook.documents existed -- which is always, after 0001 -- so the first migration after it
    (0002) was recorded as applied on both projects without its DDL ever running, and the only sign
    was one "note:" line. A migration the ledger has not seen must run."""
    if Path(filename).name != "0001_init.sql":
        return False
    cur.execute("select to_regclass('studbook.documents') is not null")
    (documents_exists,) = cur.fetchone()
    if not documents_exists:
        return False
    cur.execute(
        "insert into studbook_meta.migrations (filename, content_hash) values (%s, %s)",
        (filename, content_hash),
    )
    print(f"note: {filename}'s tables already existed with no ledger entry (see this function's "
          f"docstring) -- backfilled the ledger instead of re-running its DDL")
    return True


def ensure_login_role(cur: psycopg.Cursor, role_name: str, group_role: str, password: str) -> None:
    """Create the role if it does not exist, then (re)set its password and group membership. Safe
    to run repeatedly: never errors on an existing role, and always leaves it in the same state."""
    cur.execute("select 1 from pg_roles where rolname = %s", (role_name,))
    if cur.fetchone() is None:
        cur.execute(sql.SQL("create role {} login").format(sql.Identifier(role_name)))
    # ALTER ROLE ... PASSWORD is DDL: its grammar wants a literal token, not a bind parameter --
    # `with password %s` raises "syntax error at or near $1". sql.Literal quotes/escapes the
    # password into valid literal SQL text instead (still never string-concatenated by hand).
    cur.execute(sql.SQL("alter role {} with password {}").format(sql.Identifier(role_name), sql.Literal(password)))
    cur.execute(sql.SQL("grant {} to {}").format(sql.Identifier(group_role), sql.Identifier(role_name)))
    # The pgvector extension lives in `extensions` (0001_init.sql), not `public`, and the default
    # search_path doesn't include it -- an unqualified `vector` reference (as every client library
    # uses, e.g. pgvector-python's register_vector()) fails to resolve without this. This role-
    # level default is only a defensive fallback for a *direct* connection: confirmed live that
    # Supabase's pooler does not apply it (it reuses backend connections across logical sessions,
    # and role-level GUC defaults are only applied when Postgres starts a genuinely new backend)
    # -- every client must also `SET search_path` for itself right after connecting, pooled or
    # not (ingest/sync.py does; see its comment).
    cur.execute(sql.SQL("alter role {} set search_path = studbook, extensions").format(sql.Identifier(role_name)))


def admin_url_for_route(admin_url: str, route: Route) -> str:
    """`admin_url`, rehomed onto `route`: same user/password, host/port swapped to the route and,
    for a pooler route, the username suffixed. An earlier version of `run()` swapped host/port
    inline and forgot the username suffix, which connects but then fails with Supavisor's own
    "no tenant identifier provided" -- a bug `probe_route` could never catch, since it builds this
    URL correctly itself; only the *reuse* of the route afterward missed a field. Routed through
    this one function (and tested) so `run()` cannot make that omission again."""
    fields = conninfo_to_dict(admin_url)
    fields["host"], fields["port"] = route.host, route.port
    fields["user"] = route.username(fields["user"])
    return make_conninfo(**fields)


def connection_url(admin_url: str, route: Route, role_name: str, password: str) -> str:
    """The connection string for `role_name` over `route`: same admin credentials' other fields
    (dbname, sslmode, ...), with host/port/user/password replaced.

    Deliberately not string surgery on the URL: an earlier version of this function used
    `urllib.parse.urlsplit`/`urlunsplit`, a generic-URI parser, not a libpq one -- it chokes on a
    password containing characters libpq's URI form requires percent-encoded (`@`, `:`, `/`, `%`,
    `[`, `]` ...), and an admin password Supabase itself generates can contain them. When it
    choked, the raised `ValueError` carried a fragment of the password into its message, and that
    reached stdout -- in a live session, on 2026-09-15. `conninfo_to_dict`/`make_conninfo` are
    libpq's own parser/serialiser: no manual encoding, and regression-tested against exactly this
    failure class in tests/test_bootstrap_db.py."""
    fields = conninfo_to_dict(admin_url)
    fields["host"], fields["port"] = route.host, route.port
    fields["user"], fields["password"] = route.username(role_name), password
    return make_conninfo(**fields)


def set_user_env_var(name: str, value: str) -> None:
    """Write a Windows User environment variable via the registry (what `setx` and the System
    Properties GUI both do), then broadcast WM_SETTINGCHANGE so new processes pick it up without
    a reboot. Keeps the value out of any child process's command line or environment block.

    Windows only, and imported here rather than at module level: the first CI run, on Linux, could
    not even import this module for its unit tests (docs/ci.md). Elsewhere it refuses, naming what
    to do, rather than falling back to printing a password or writing it to a file."""
    if sys.platform != "win32":
        raise RuntimeError(
            f"cannot store {name}: bootstrap_db.py keeps generated passwords in Windows user "
            "environment variables, and this is not Windows. Set the role's password in your own "
            "secret store and export it yourself; no password was printed or written."
        )
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x001A, 0x0002
    result = ctypes.c_long()
    ctypes.windll.user32.SendMessageTimeoutW(
        HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment", SMTO_ABORTIFHUNG, 5000, ctypes.byref(result)
    )


def run(admin_url: str, roles: list[tuple[str, str, str]] = ROLES, rotate_roles: bool = True) -> None:
    route = probe_route(admin_url)
    routed_admin_url = admin_url_for_route(admin_url, route)

    with connect(routed_admin_url, autocommit=False) as conn, conn.cursor() as cur:
        ensure_migrations_table(cur)
        conn.commit()

        applied, skipped = 0, 0
        for migration in MIGRATIONS:
            rel = str(migration.relative_to(ROOT))
            text = migration.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            cur.execute("select content_hash from studbook_meta.migrations where filename = %s", (rel,))
            row = cur.fetchone()
            if row is not None:
                if row[0] != content_hash:
                    raise SystemExit(
                        f"{rel} was already applied with different content than the file on disk "
                        f"now -- a migration must never be edited after it is applied; add a new one"
                    )
                print(f"skip {rel} (already applied)")
                skipped += 1
                continue

            if backfill_if_already_applied(cur, rel, content_hash):
                conn.commit()
                skipped += 1
                continue

            print(f"applying {rel} ...")
            cur.execute(text)
            cur.execute(
                "insert into studbook_meta.migrations (filename, content_hash) values (%s, %s)",
                (rel, content_hash),
            )
            conn.commit()
            applied += 1
        print(f"schema: {applied} migration(s) applied, {skipped} already applied")

        # Rotating a role's password changes its connection string, and copies of that string live
        # outside this machine's env vars -- Paddock stores the reader's encrypted in its own
        # settings. So a schema change goes out with rotate_roles=False (--migrations-only), which
        # leaves every role and every stored connection string exactly as it was.
        for role_name, group_role, env_var in roles if rotate_roles else []:
            password = secrets.token_urlsafe(32)
            ensure_login_role(cur, role_name, group_role, password)
            set_user_env_var(env_var, connection_url(admin_url, route, role_name, password))
            print(f"role: {role_name} (member of {group_role}) -> {env_var} (value not shown)")
        conn.commit()

    with connect(routed_admin_url, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("select tablename, rowsecurity from pg_tables where schemaname = 'studbook' order by tablename")
        print("\nRLS status:")
        for tablename, rowsecurity in cur.fetchall():
            print(f"  studbook.{tablename}: rowsecurity={rowsecurity}")
        cur.execute(
            "select schemaname, tablename, policyname, roles, cmd from pg_policies"
            " where schemaname = 'studbook' order by tablename, policyname"
        )
        print("\nPolicies:")
        for schemaname, tablename, policyname, policy_roles, cmd in cur.fetchall():
            print(f"  {schemaname}.{tablename} {policyname}: cmd={cmd} roles={policy_roles}")

    if not rotate_roles:
        print("\nDone. Migrations only: no role was touched and no connection string changed.")
        return
    names = " and ".join(env_var for _, _, env_var in roles)
    print(
        f"\nDone. {names} are set as Windows User environment variables (passwords freshly "
        "rotated, not printed here). This process and any shell open before this run will not see "
        "them -- restart the session to pick them up."
    )


def main() -> int:
    args = sys.argv[1:]
    target = "test" if "--target=test" in args or any(
        a == "--target" and args[i + 1:i + 2] == ["test"] for i, a in enumerate(args)) else "production"
    rotate_roles = "--migrations-only" not in args
    admin_env, roles = TARGETS[target]
    admin_url = os.environ.get(admin_env)
    if not admin_url:
        print(f"{admin_env} is not set in this process's environment.", file=sys.stderr)
        return 2
    refuse_production(target, admin_url, os.environ.get("STUDBOOK_DATABASE_URL"))
    if rotate_roles:
        print(f"target: {target} (role connection strings go to {', '.join(e for _, _, e in roles)})")
    else:
        print(f"target: {target}, migrations only (no role or connection string is touched)")
    try:
        run(admin_url, roles, rotate_roles)
    except SystemExit:
        raise
    except psycopg.Error as e:
        # Safe to print in full: libpq's own errors (auth, syntax, constraint violations,
        # connection drops) never echo credentials back -- that class of leak was specifically in
        # the generic string parser this module no longer uses (see connection_url's docstring).
        print(f"failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 -- last-resort guard for anything NOT from psycopg/libpq
        # (a bug in this module's own code, say): withhold the message on principle, since this
        # boundary is exactly the one that leaked a secret once before, and an unrecognised
        # exception type has no established guarantee about what it might contain.
        print(f"failed: {type(e).__name__} (message withheld; re-run with --debug to see it)",
              file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
