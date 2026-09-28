// Why do the reranker logits differ between sentence-transformers and the ONNX port on some
// passages, while the ranking mostly agrees? For each question named on the command line: every
// fused candidate with its Python logit, its JS logit, and its token count under the JS tokenizer
// -- the hypothesis being that only passages past the 512-token window diverge.
//
//   node diagnose_rerank.mjs version-16 wh-07
import { readFileSync } from 'node:fs';
import { AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';
import pg from 'pg';

const ids = process.argv.slice(2);
const python = JSON.parse(readFileSync(new URL('./python_side.json', import.meta.url), 'utf8'));
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bge-reranker-base');
const model = await AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base', { dtype: 'fp32' });
console.log(`tokenizer model_max_length: ${tokenizer.model_max_length}`);

const text = process.env.STUDBOOK_DATABASE_URL;
const kv = Object.fromEntries([...text.matchAll(/(\w+)\s*=\s*(?:'((?:\\.|[^'])*)'|(\S+))/g)].map((m) => [m[1], m[2] ?? m[3]]));
const ca = readFileSync(new URL('./supabase-ca.pem', import.meta.url), 'utf8');
const client = new pg.Client({ host: kv.host, port: Number(kv.port), user: kv.user, password: kv.password, database: kv.dbname, ssl: { rejectUnauthorized: true, ca } });
await client.connect();

for (const id of ids) {
  const q = python.find((r) => r.id === id);
  const { rows } = await client.query('select id, body from studbook.chunks where id = any($1)', [q.fused_ids]);
  const body = Object.fromEntries(rows.map((r) => [r.id, r.body]));
  const passages = q.fused_ids.map((c) => body[c]);
  const inputs = tokenizer(passages.map(() => q.question), { text_pair: passages, padding: true, truncation: true });
  const { logits } = await model(inputs);
  const js = logits.tolist().map((r) => r[0]);
  const pyLogit = Object.fromEntries(q.reranked.map((r) => [r.chunk_id, r.logit]));
  console.log(`\n${id}: ${q.question}`);
  console.log(`  ${'chunk'.padEnd(52)} ${'py'.padStart(8)} ${'js'.padStart(8)} ${'diff'.padStart(7)} ${'pair tokens'.padStart(12)}`);
  q.fused_ids.forEach((c, i) => {
    const pairTokens = tokenizer(q.question, { text_pair: passages[i] }).input_ids.dims[1];
    const py = pyLogit[c];
    const diff = py === undefined ? '' : (js[i] - py).toFixed(2);
    console.log(`  ${c.slice(-52).padEnd(52)} ${(py === undefined ? '(not top10)' : py.toFixed(2)).padStart(8)} ${js[i].toFixed(2).padStart(8)} ${diff.padStart(7)} ${String(pairTokens).padStart(12)}${pairTokens > 512 ? '  > 512' : ''}`);
  });
}
await client.end();
