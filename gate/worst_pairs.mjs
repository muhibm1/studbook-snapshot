// Across every dev question: the (question, passage) pairs where the ONNX reranker's probability
// differs most from sentence-transformers', with token counts, plus the two top-5 lists for any
// question whose top-5 differs, gold-marked -- so a disagreement can be read as "which passage,
// how long, and did it matter".
import { readFileSync } from 'node:fs';
import { AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';
import pg from 'pg';

const python = JSON.parse(readFileSync(new URL('./python_side.json', import.meta.url), 'utf8'));
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bge-reranker-base');
const model = await AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base', { dtype: 'fp32' });
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

const text = process.env.STUDBOOK_DATABASE_URL;
const kv = Object.fromEntries([...text.matchAll(/(\w+)\s*=\s*(?:'((?:\\.|[^'])*)'|(\S+))/g)].map((m) => [m[1], m[2] ?? m[3]]));
const ca = readFileSync(new URL('./supabase-ca.pem', import.meta.url), 'utf8');
const client = new pg.Client({ host: kv.host, port: Number(kv.port), user: kv.user, password: kv.password, database: kv.dbname, ssl: { rejectUnauthorized: true, ca } });
await client.connect();
const allIds = [...new Set(python.flatMap((q) => q.fused_ids))];
const { rows } = await client.query('select id, body from studbook.chunks where id = any($1)', [allIds]);
await client.end();
const body = Object.fromEntries(rows.map((r) => [r.id, r.body]));

const diffs = [];
const disagreements = [];
for (const q of python) {
  const passages = q.fused_ids.map((c) => body[c]);
  const inputs = tokenizer(passages.map(() => q.question), { text_pair: passages, padding: true, truncation: true });
  const { logits } = await model(inputs);
  const js = logits.tolist().map((r) => sigmoid(r[0]));
  const jsProb = Object.fromEntries(q.fused_ids.map((c, i) => [c, js[i]]));
  for (const r of q.reranked) {
    const tokens = tokenizer(q.question, { text_pair: body[r.chunk_id] }).input_ids.dims[1];
    diffs.push({ id: q.id, chunk: r.chunk_id, py: r.logit, js: jsProb[r.chunk_id], tokens });
  }
  const jsTop5 = q.fused_ids.map((c, i) => [c, js[i]]).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const pyTop5 = q.reranked.slice(0, 5);
  const same = jsTop5.every(([c]) => pyTop5.some((p) => p.chunk_id === c));
  if (!same) disagreements.push({ q, jsTop5, pyTop5 });
}

diffs.sort((a, b) => Math.abs(b.js - b.py) - Math.abs(a.js - a.py));
console.log('worst 8 pairs by |js prob - py prob|:');
for (const d of diffs.slice(0, 8)) {
  console.log(`  ${d.id.padEnd(11)} py=${d.py.toFixed(4)} js=${d.js.toFixed(4)} diff=${(d.js - d.py).toFixed(4)}  tokens=${String(d.tokens).padStart(5)}${d.tokens > 512 ? ' >512' : ''}  ${d.chunk.slice(-45)}`);
}
const over = diffs.filter((d) => d.tokens > 512), under = diffs.filter((d) => d.tokens <= 512);
const maxAbs = (xs) => Math.max(...xs.map((d) => Math.abs(d.js - d.py)));
console.log(`\nmax |diff| for pairs <= 512 tokens: ${maxAbs(under).toFixed(6)} (n=${under.length})`);
console.log(`max |diff| for pairs  > 512 tokens: ${maxAbs(over).toFixed(6)} (n=${over.length})`);

for (const { q, jsTop5, pyTop5 } of disagreements) {
  const gold = (c) => (q.gold_quotes.some((g) => body[c].includes(g)) ? ' [GOLD]' : '');
  console.log(`\n${q.id} top-5 differs`);
  console.log('  python: ' + pyTop5.map((p) => `${p.chunk_id.slice(-30)}=${p.logit.toFixed(4)}${gold(p.chunk_id)}`).join('  '));
  console.log('  js:     ' + jsTop5.map(([c, s]) => `${c.slice(-30)}=${s.toFixed(4)}${gold(c)}`).join('  '));
}
