# CI, and what it deliberately does not do

`.github/workflows/ci.yml`, added 2026-09-18. Two jobs, because what may run on every push and what
may touch a database are different questions.

## `checks` — every push and pull request, no credentials

| step | what it proves |
|---|---|
| `ruff check --select F,E9 .` | no undefined names, unused imports or syntax errors. Its first run found a real bug: `eval/ceiling.py` printed an undefined name on its last line, so it computed everything and then crashed whenever any question's gold passage was never retrieved (it always is on dev: `version-06`, `version-14`). |
| unit suite | 181 tests run without credentials; the 30 live tests skip. The job **asserts at least 150 ran**, so a suite that skipped its way to green fails. |
| `pip-audit` | no known vulnerability in any pinned dependency, except eight accepted by id below. A new one fails the build. |

Every action is pinned to a full commit sha (a tag can be moved after review; a sha cannot), and
`tests/test_ci_workflow.py` fails if one is not, if a workflow stops parsing, or if either job's
secrets drift from what this page promises.

The first Linux run found a second real bug: `store/bootstrap_db.py` imported the Windows-only
`winreg` at module level, so on any other platform the module could not be imported at all — the
same script this page tells you to run to set up a test project. It now imports it where a
password is stored, and off Windows refuses with a message instead of falling back to printing or
writing the password anywhere.

Not in CI, on purpose: `python -m eval.gold check`, which needs the corpus repositories' working
trees. Run it locally before any change to the gold set.

## `live` — by hand only, against a dedicated test project

Dispatch the workflow with **live** ticked. It runs the full suite and fails if a single test
skips. It reads `STUDBOOK_TEST_DATABASE_URL`, `STUDBOOK_TEST_INGEST_DATABASE_URL` and
`STUDBOOK_TEST_ANTHROPIC_API_KEY` and nothing else. There is deliberately **no fallback to the
production connection strings**: the RLS suite writes probe rows, and a suite that creates and
checks data will eventually run against whatever its variables name. With a secret missing, the job
fails and names it.

**The test project** (set up 2026-09-19): `studbook-test` (ref `testprojectrefabcdef`), a separate
free-plan Supabase project in us-east-1, never the production one (`productionprojectref`). To
rebuild it:

1. Create a project and store its `postgres` connection string as the Windows user variable
   `STUDBOOK_TEST_ADMIN_DATABASE_URL`.
2. `.venv/Scripts/python.exe -m store.bootstrap_db --target test` applies the migrations and creates
   the two login roles, writing their connection strings to `STUDBOOK_TEST_DATABASE_URL` and
   `STUDBOOK_TEST_INGEST_DATABASE_URL` only. **Without `--target test` it writes the production
   variables** -- which is what the first version of this page would have had you do. The test
   target refuses outright if its admin connection names the production project.
3. Ingest the corpus with `STUDBOOK_INGEST_DATABASE_URL` set to the *test* ingest value for that one
   process (`python -m ingest.sync --all`): 175 documents, 530 passages on 2026-09-19.
4. Set the three repository secrets from those variables with `gh secret set` over stdin. The
   Anthropic key is currently the production key, by the owner's decision; a separate key with a
   spend limit is the better setting when there is time.

## The dependency upgrade CI forced

The first audit found 66 advisories: 23 against torch 2.4.1, 43 against transformers 4.42.4, both
pinned old on purpose (`requirements.txt`). Re-tested rather than accepted:

- **transformers 4.42.4 → 5.17.0, sentence-transformers 3.0.1 → 6.1.0.** The import-time regression
  that held them back no longer reproduces. All 43 transformers advisories closed.
- **torch 2.4.1 → 2.8.0**, the newest build that loads on the development machine — bisected: 2.9.0
  and every later build fail loading `c10.dll` with WinError 1114. 15 of torch's 23 closed.
- **Outputs are identical**: maximum embedding difference 0.0 and maximum rerank-score difference
  1.6e-10 on the same inputs across the two stacks, and the dev retrieval eval reproduces every
  published number to four places (recall@5 0.8529, coverage@5 0.7941, MRR 0.6385, precision
  0.1882). No stored embedding needed re-computing, and Paddock's parity gates hold.

To re-run that comparison before any future bump: `gate/numerics.py` in each environment, then
`--compare`; then the retrieval eval and both Paddock gates (re-run after this upgrade: both pass).

## The eight accepted torch advisories

Each needs torch 2.9.0 or later, which does not load on the development machine. Studbook uses
torch only for inference on two fixed BERT-family models. "Never called" is **confirmed for
Studbook's own code** (none of these functions appears in it, by grep) and **believed, not
verified, for the libraries' BERT forward pass**, which is attention and linear layers.

| id | what | why it is not reachable here |
|---|---|---|
| PYSEC-2025-203 | `torch.linalg.lu` denial of service | never called |
| PYSEC-2025-204 | `torch.rot90` with `randn_like` misbehaves | never called |
| PYSEC-2025-206 | integer overflow in `nan_to_num(...).long()` | never called |
| PYSEC-2025-193 | memory corruption in `nn.utils.rnn.unpack_sequence` | no RNNs; both models are transformers |
| PYSEC-2025-195 | memory corruption in `torch.lstm_cell` | no LSTMs |
| PYSEC-2025-194 | memory corruption in `torch.jit.script` | never scripted |
| PYSEC-2026-139 | deserialization in the `.pt2` loader (no fix released) | no `.pt2` files are loaded |
| PYSEC-2026-2286 | a crafted `.pth` can escape the `weights_only` unpickler | **made unreachable by construction**: both models load with `use_safetensors=True`, so a pickle checkpoint is refused rather than unpickled, and at a pinned upstream revision, so a changed repository cannot introduce one. `tests/test_model_pins.py` fails if either guard is dropped. |

What would close them properly: a torch build that loads on this machine (a newer Windows, a
different machine, or running the service on Linux, where CI already installs torch 2.8.0 and
could take a newer one), then removing each id from the workflow as it stops being reported.
Review this table whenever torch is bumped.

## Branch protection

On since 2026-09-18 (GitHub Pro), read back from the API after setting:

| rule | setting |
|---|---|
| required check before a pull request merges | `checks` (the job above), branch up to date with `main` first |
| force pushes to `main` | refused |
| deleting `main` | refused |
| admins | may push directly; GitHub records each such push as bypassing the rule |

The last row is a deliberate choice for a sole committer, and one setting away from binding admins
too (`enforce_admins`), at the cost of a pull request per change. Dependabot's pull requests cannot
merge red.

`.githooks/pre-push` stays: it runs ruff and the unit suite (credentials removed) before a push
leaves the machine, which catches a red push before GitHub has to record a bypass for it. Enable it
once per clone with `git config core.hooksPath .githooks`.
