// Node side of the parity gate (docs/paddock-gate.md). Reproduces Studbook's query path with the
// ONNX ports of the same two models, the same SQL and the same fusion, then holds the result to
// the Python pipeline: per-question, against gate/python_side.json, and in aggregate against the
// published dev numbers (recall@5 0.8235, mrr@10 0.6485, precision@5 0.1824).
//
//   set STUDBOOK_DATABASE_URL, then:  npm run gate
//
// Nothing here is the Paddock port. It is the evidence for whether the port is safe to write.

import { readFileSync } from 'node:fs';
import { pipeline, AutoTokenizer, AutoModelForSequenceClassification, Tensor } from '@huggingface/transformers';
import pg from 'pg';

const EMBED_MODEL = 'Xenova/bge-base-en-v1.5';
const RERANK_MODEL = 'Xenova/bge-reranker-base';
const ARM_K = 20, RRF_K = 10, TOP_K = 5, MRR_K = 10;
const PUBLISHED = { recall5: 0.8235, mrr10: 0.6485, precision5: 0.1824 };

const url = process.env.STUDBOOK_DATABASE_URL;
if (!url) { console.error('STUDBOOK_DATABASE_URL is not set'); process.exit(2); }
const python = JSON.parse(readFileSync(new URL('./python_side.json', import.meta.url), 'utf8'));

// ---------------------------------------------------------------- models (fp32, never quantised)
console.log('loading models ...');
const embed = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'fp32' });
const tokenizer = await AutoTokenizer.from_pretrained(RERANK_MODEL);
const reranker = await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, { dtype: 'fp32' });

async function embedQueries(texts) {
  // CLS pooling + L2 normalisation is what sentence-transformers does for this model.
  const out = await embed(texts, { pooling: 'cls', normalize: true });
  return out.tolist();
}

// Truncation is done here, by token id, not by the tokenizer's `truncation: true`. On pairs
// over 512 tokens (46% of what the reranker sees on this corpus) transformers.js's pair
// truncation diverges from HuggingFace's `longest_first`, which sentence-transformers uses:
// probabilities differed by up to 0.094 and two top-5s flipped. Keeping the first
// 512 - specials - |query| passage tokens and assembling <s> q </s></s> p </s> ourselves
// reproduces the Python scores to 4e-6 on every one of 157 long pairs (gate/truncation.mjs,
// strategies A/B/C). Keeping the passage's tail instead is off by up to 0.97, so this is not a
// detail the port can leave to a library default.
const MAX_TOKENS = 512;
// int64 tensors list as BigInt; everything downstream wants plain numbers.
const bareIds = (s) => tokenizer(s, { add_special_tokens: false }).input_ids.tolist()[0].map(Number);
// Special-token ids read off real encodings rather than tokenizer properties, which
// transformers.js does not expose as `cls_token_id` etc. A one-token pair encodes as
// <s> a </s></s> b </s>, so positions 0 and 2 are CLS and SEP; a padded two-item batch
// exposes PAD at the short item's tail.
const probe = tokenizer('a', { text_pair: 'b' }).input_ids.tolist()[0].map(Number);
const CLS = probe[0], SEP = probe[2];
const SPECIALS = probe.length - 2;
const padded = tokenizer(['a', 'a a a a'], { padding: true }).input_ids.tolist().map((r) => r.map(Number));
const PAD = padded[0][padded[0].length - 1];
if (![CLS, SEP, PAD].every(Number.isInteger)) throw new Error(`special token ids not resolved: ${[CLS, SEP, PAD]}`);

async function rerankLogits(query, passages) {
  const q = bareIds(query);
  const budget = MAX_TOKENS - SPECIALS - q.length;
  const seqs = passages.map((p) => {
    const ids = bareIds(p);
    const kept = ids.length <= budget ? ids : ids.slice(0, budget);
    return [CLS, ...q, SEP, SEP, ...kept, SEP];
  });
  const width = Math.max(...seqs.map((s) => s.length));
  const pad = PAD;
  const flatIds = [], flatMask = [];
  for (const s of seqs) {
    flatIds.push(...s, ...Array(width - s.length).fill(pad));
    flatMask.push(...s.map(() => 1), ...Array(width - s.length).fill(0));
  }
  const input_ids = new Tensor('int64', BigInt64Array.from(flatIds.map(BigInt)), [seqs.length, width]);
  const attention_mask = new Tensor('int64', BigInt64Array.from(flatMask.map(BigInt)), [seqs.length, width]);
  const { logits } = await reranker({ input_ids, attention_mask });
  return logits.tolist().map((row) => row[0]);
}

// ---------------------------------------------------------------- retrieval (same SQL as retrieve/hybrid.py)
// STUDBOOK_DATABASE_URL is a libpq keyword/value conninfo ("user=... password=... host=..."),
// written that way by store/bootstrap_db.py because it survives passwords with URL-hostile
// characters. psycopg parses it natively; node-postgres only takes URLs, and fed a conninfo it
// resolved the host as "base" (from "database"). Parse it the way libpq does: whitespace-
// separated key=value, values optionally single-quoted with backslash escapes.
function parseConninfo(text) {
  if (/^postgres(ql)?:\/\//.test(text)) return { connectionString: text };
  const options = {};
  const re = /(\w+)\s*=\s*(?:'((?:\\.|[^'])*)'|(\S+))/g;
  for (const m of text.matchAll(re)) {
    options[m[1]] = (m[2] ?? m[3]).replace(/\\(.)/g, '$1');
  }
  const map = { dbname: 'database' };
  const out = {};
  for (const [k, v] of Object.entries(options)) out[map[k] ?? k] = k === 'port' ? Number(v) : v;
  return out;
}
// Verification is ON, against Supabase's own root. The pooler presents a private PKI
// ("Supabase Root 2021 CA" -> intermediate -> *.pooler.supabase.com), which Node's public CA
// store rightly rejects as "self-signed certificate in certificate chain". gate/extract_ca.mjs
// saved that root from the handshake (trust-on-first-use: cross-check its subject and SHA-256
// against the CA certificate the Supabase dashboard offers for download). A finding for the
// Python side too: psycopg's default sslmode=prefer encrypts without verifying, so every
// connection this project has made so far was encrypted but never authenticated. It should
// move to sslmode=verify-full with this same CA.
const ca = readFileSync(new URL('./supabase-ca.pem', import.meta.url), 'utf8');
const client = new pg.Client({ ...parseConninfo(url), ssl: { rejectUnauthorized: true, ca } });
await client.connect();
await client.query('set search_path to studbook, extensions');

const SELECT = 'select c.id as chunk_id, c.document_id, c.heading_path, c.body, d.repo, d.doc_type ' +
  'from studbook.chunks c join studbook.documents d on d.id = c.document_id';

async function retrieveVector(embedding, k) {
  const { rows } = await client.query(
    `${SELECT} where true order by c.embedding <=> $1::vector limit $2`,
    [`[${embedding.join(',')}]`, k],
  );
  return rows;
}

async function retrieveFulltext(text, k) {
  const { rows } = await client.query(
    'with q as (select to_tsquery(\'english\', string_agg(lexeme, \' | \')) as tsq ' +
    '  from unnest(tsvector_to_array(to_tsvector(\'english\', $1))) as lexeme) ' +
    `${SELECT}, q where q.tsq is not null and c.fts @@ q.tsq ` +
    'order by ts_rank(c.fts, q.tsq) desc limit $2',
    [text, k],
  );
  return rows;
}

function rrf(rankings, k) {
  const scores = new Map(), byId = new Map();
  for (const ranking of rankings) {
    ranking.forEach((r, i) => {
      scores.set(r.chunk_id, (scores.get(r.chunk_id) ?? 0) + 1 / (k + i + 1));
      byId.set(r.chunk_id, r);
    });
  }
  return [...scores.keys()].sort((a, b) => scores.get(b) - scores.get(a)).map((id) => byId.get(id));
}

// ---------------------------------------------------------------- run it
const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const isHit = (row, quotes) => quotes.some((q) => row.body.includes(q));

const embeddings = await embedQueries(python.map((q) => q.question));
const totals = { n: 0, recall: 0, mrr: 0, precision: 0 };
const parity = { cosMin: 1, cosSum: 0, fusedExact: 0, top5Same: 0, logitMaxDiff: 0, vectorExact: 0, fulltextExact: 0 };

for (const [i, q] of python.entries()) {
  const embedding = embeddings[i];
  const cos = cosine(embedding, q.embedding);
  parity.cosMin = Math.min(parity.cosMin, cos); parity.cosSum += cos;

  const vector = await retrieveVector(embedding, ARM_K);
  const fulltext = await retrieveFulltext(q.question, ARM_K);
  const fused = rrf([vector, fulltext], RRF_K).slice(0, ARM_K);
  const logits = await rerankLogits(q.question, fused.map((r) => r.body));
  const order = fused.map((_, j) => j).sort((a, b) => logits[b] - logits[a]);
  const reranked = order.slice(0, MRR_K).map((j) => fused[j]);

  // per-question parity with the Python run
  const same = (xs, ys) => xs.length === ys.length && xs.every((x, j) => x === ys[j]);
  parity.vectorExact += same(vector.map((r) => r.chunk_id), q.vector_ids);
  parity.fulltextExact += same(fulltext.map((r) => r.chunk_id), q.fulltext_ids);
  parity.fusedExact += same(fused.map((r) => r.chunk_id), q.fused_ids);
  // sentence-transformers' CrossEncoder.predict() applies a sigmoid to this single-label model;
  // the ONNX head returns the raw logit. Compare probabilities to probabilities.
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const pyTop5 = new Set(q.reranked.slice(0, TOP_K).map((r) => r.chunk_id));
  const jsTop5 = new Set(reranked.slice(0, TOP_K).map((r) => r.chunk_id));
  const top5Same = [...jsTop5].every((id) => pyTop5.has(id)) && jsTop5.size === pyTop5.size;
  parity.top5Same += top5Same;
  for (const r of q.reranked) {
    const j = fused.findIndex((f) => f.chunk_id === r.chunk_id);
    if (j >= 0) parity.logitMaxDiff = Math.max(parity.logitMaxDiff, Math.abs(sigmoid(logits[j]) - r.logit));
  }
  if (!top5Same) {
    // How close was the swap? The gap between the 5th and 6th JS scores is the margin the two
    // runtimes disagreed across; a gap inside kernel noise is a tie, not a divergence.
    const sortedProbs = order.map((j) => sigmoid(logits[j]));
    parity.tieGaps ??= [];
    parity.tieGaps.push({ id: q.id, gap: sortedProbs[TOP_K - 1] - sortedProbs[TOP_K], fifth: sortedProbs[TOP_K - 1] });
  }

  // the published metrics, computed exactly as eval/metrics.py does
  const hits = reranked.map((r) => isHit(r, q.gold_quotes));
  const first = hits.indexOf(true);
  totals.n += 1;
  totals.recall += hits.slice(0, TOP_K).some(Boolean) ? 1 : 0;
  totals.mrr += first >= 0 ? 1 / (first + 1) : 0;
  totals.precision += hits.slice(0, TOP_K).filter(Boolean).length / TOP_K;
  process.stdout.write(`  ${q.id.padEnd(12)} cos=${cos.toFixed(6)}  fused=${same(fused.map((r) => r.chunk_id), q.fused_ids) ? 'same' : 'DIFF'}  top5=${[...jsTop5].every((id) => pyTop5.has(id)) ? 'same' : 'DIFF'}\n`);
}
await client.end();

const n = totals.n;
const got = { recall5: totals.recall / n, mrr10: totals.mrr / n, precision5: totals.precision / n };
console.log(`\nembedding parity (n=${n}): cosine min ${parity.cosMin.toFixed(6)}, mean ${(parity.cosSum / n).toFixed(6)}`);
console.log(`vector arm identical:    ${parity.vectorExact}/${n}`);
console.log(`fulltext arm identical:  ${parity.fulltextExact}/${n}`);
console.log(`fused top-20 identical:  ${parity.fusedExact}/${n}`);
console.log(`reranked top-5 identical: ${parity.top5Same}/${n}   (max |probability diff| ${parity.logitMaxDiff.toFixed(6)})`);
for (const t of parity.tieGaps ?? []) {
  console.log(`  ${t.id}: 5th-place score ${t.fifth.toFixed(5)}, gap to 6th ${t.gap.toFixed(6)} -- ${t.gap < 1e-3 ? 'a tie inside kernel noise' : 'a real disagreement'}`);
}
console.log('\nmetric        js       python');
for (const k of Object.keys(PUBLISHED)) {
  console.log(`  ${k.padEnd(11)} ${got[k].toFixed(4)}   ${PUBLISHED[k].toFixed(4)}   ${Math.abs(got[k] - PUBLISHED[k]) < 5e-5 ? 'match' : 'DIFFERS'}`);
}
