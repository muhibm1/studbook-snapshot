const $ = (id) => document.getElementById(id);
let repoData = [];

// The key lives in sessionStorage, not localStorage: it dies with the tab, so a shared machine
// never keeps it, and nothing here ever writes it anywhere else. Every read and write is guarded --
// storage can be unavailable in a private window.
const KEY_SLOT = 'studbook-api-key';
function loadKey() { try { return sessionStorage.getItem(KEY_SLOT) || ''; } catch { return ''; } }
function saveKey(v) { try { sessionStorage.setItem(KEY_SLOT, v); } catch { /* fine: ask again next tab */ } }
function authHeaders(extra = {}) {
  const key = $('key').value.trim();
  return key ? { ...extra, authorization: `Bearer ${key}` } : extra;
}

$('key').value = loadKey();
let loadedWithKey = null;  // the key the scope filter was last built with, so a blur without a change is a no-op
$('key').addEventListener('change', () => { saveKey($('key').value.trim()); loadRepos(); });

// Touches the DOM only after the fetch resolves. This handler runs on the key field's `change`,
// which fires on blur -- including the blur caused by pressing Ask -- and rebuilding the selects
// synchronously in that moment re-laid-out the form under the pointer and ate the click. Nothing
// here mutates anything until the response is back, by which time the click has long dispatched.
async function loadRepos() {
  const key = $('key').value.trim();
  if (key === loadedWithKey) return;
  let repos = [];
  try {
    const r = await fetch('/repos', { headers: authHeaders() });
    if (r.ok) repos = (await r.json()).repos || [];
  } catch { /* scope filter is optional; the page works without it */ }
  loadedWithKey = key;
  repoData = repos;
  $('repo').replaceChildren(new Option('All repositories', ''), ...repos.map(({ repo }) => new Option(repo, repo)));
  $('change').replaceChildren(new Option('All changes', ''));
}

$('repo').addEventListener('change', () => {
  const sel = repoData.find((r) => r.repo === $('repo').value);
  $('change').replaceChildren(new Option('All changes', ''));
  for (const c of sel?.changes ?? []) $('change').append(new Option(c.change_id, c.change_id));
});

// The model writes citations as [1], [2]. Render them as clickable chips that open the matching
// receipt, escaping everything else so answer text can never inject markup.
function renderAnswer(text) {
  const frag = document.createDocumentFragment();
  for (const para of text.split(/\n\n+/)) {
    const p = document.createElement('p');
    let last = 0;
    for (const m of para.matchAll(/\[(\d+)\]/g)) {
      p.append(para.slice(last, m.index));
      const chip = document.createElement('cite-ref');
      chip.textContent = m[1];
      chip.onclick = () => {
        const d = $('r' + m[1]);
        if (d) { d.open = true; d.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
      };
      p.append(chip);
      last = m.index + m[0].length;
    }
    p.append(para.slice(last));
    frag.append(p);
  }
  return frag;
}

function render(data) {
  const out = $('out');
  out.replaceChildren();

  const box = document.createElement('div');
  box.className = 'answer' + (data.refused ? ' refused' : '');
  if (data.refused) {
    const b = document.createElement('div');
    b.className = 'badge';
    b.textContent = 'not in the record';
    box.append(b);
  }
  box.append(renderAnswer(data.answer));

  const cited = new Set(data.citations.map((c) => c.number));
  const shown = data.passages.filter((p) => cited.has(p.number));
  const rest = data.passages.filter((p) => !cited.has(p.number));

  const rec = document.createElement('div');
  rec.className = 'receipts';
  const h = document.createElement('h2');
  h.textContent = shown.length ? `Receipts (${shown.length} cited)` : 'Passages retrieved';
  rec.append(h);
  for (const p of [...shown, ...rest]) {
    const d = document.createElement('details');
    d.id = 'r' + p.number;
    const s = document.createElement('summary');
    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = p.number;
    const where = document.createElement('span');
    where.className = 'where';
    where.textContent = `${p.repo} / ${p.doc_type} — ${p.heading_path}`;
    s.append(n, where);
    const pre = document.createElement('pre');
    pre.className = 'chunk';
    pre.textContent = p.body;
    d.append(s, pre);
    rec.append(d);
  }
  box.append(rec);

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${data.model} · ${data.input_tokens} in / ${data.output_tokens} out`;
  box.append(meta);
  out.append(box);
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = $('q').value.trim();
  if (!question) return;
  $('go').disabled = true;
  $('go').textContent = 'Asking…';
  $('out').replaceChildren(Object.assign(document.createElement('p'), {
    className: 'hint', textContent: 'Retrieving and reading the record…',
  }));
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: authHeaders({ 'content-type': 'application/json' }),
      body: JSON.stringify({
        question,
        repo: $('repo').value || null,
        change_id: $('change').value || null,
      }),
    });
    if (r.status === 401) {
      $('key').focus();
      throw new Error($('key').value.trim()
        ? 'API key rejected. Check it against STUDBOOK_API_KEY on the server.'
        : 'Enter the API key (STUDBOOK_API_KEY on the server) to ask.');
    }
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    render(await r.json());
  } catch (err) {
    $('out').replaceChildren(Object.assign(document.createElement('p'), {
      className: 'err', textContent: String(err.message || err),
    }));
  } finally {
    $('go').disabled = false;
    $('go').textContent = 'Ask';
  }
});

loadRepos();
