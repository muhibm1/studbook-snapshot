// Which truncation reproduces sentence-transformers' CrossEncoder on pairs over 512 tokens?
// Hypotheses, each scored on every >512 pair against the Python probabilities in python_side.json:
//   A  transformers.js default (`truncation: true` on the pair)        -- known to diverge
//   B  keep the FIRST n passage tokens, n = 512 - query tokens - specials  (HF longest_first)
//   C  keep the LAST n passage tokens
// The winner is whatever drives max |diff| on long pairs down to the <= 512 level (~4e-6).
import { readFileSync } from 'node:fs';
import { AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';
import pg from 'pg';

const python = JSON.parse(readFileSync(new URL('./python_side.json', import.meta.url), 'utf8'));
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bge-reranker-base');
const model = await AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base', { dtype: 'fp32' });
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const MAX = 512;

const text = process.env.STUDBOOK_DATABASE_URL;
const kv = Object.fromEntries([...text.matchAll(/(\w+)\s*=\s*(?:'((?:\\.|[^'])*)'|(\S+))/g)].map((m) => [m[1], m[2] ?? m[3]]));
const ca = readFileSync(new URL('./supabase-ca.pem', import.meta.url), 'utf8');
const client = new pg.Client({ host: kv.host, port: Number(kv.port), user: kv.user, password: kv.password, database: kv.dbname, ssl: { rejectUnauthorized: true, ca } });
await client.connect();
const ids = [...new Set(python.flatMap((q) => q.reranked.map((r) => r.chunk_id)))];
const { rows } = await client.query('select id, body from studbook.chunks where id = any($1)', [ids]);
await client.end();
const body = Object.fromEntries(rows.map((r) => [r.id, r.body]));

// Token ids for a bare sequence, without the special tokens the pair encoding adds.
const bare = (s) => tokenizer(s, { add_special_tokens: false }).input_ids.tolist()[0];
// XLM-R pair layout: <s> A </s></s> B </s>  -> 4 special tokens.
const SPECIALS = tokenizer('a', { text_pair: 'b' }).input_ids.dims[1] - 2;

async function score(query, passage, strategy) {
  if (strategy === 'A') {
    const inputs = tokenizer(query, { text_pair: passage, truncation: true });
    return sigmoid((await model(inputs)).logits.tolist()[0][0]);
  }
  const q = bare(query), p = bare(passage);
  const budget = MAX - SPECIALS - q.length;
  const kept = p.length <= budget ? p : (strategy === 'B' ? p.slice(0, budget) : p.slice(p.length - budget));
  // Rebuild the pair from token ids so no re-tokenisation of decoded text can shift a boundary.
  const input_ids = [tokenizer.cls_token_id ?? 0, ...q, tokenizer.sep_token_id, tokenizer.sep_token_id, ...kept, tokenizer.sep_token_id];
  const inputs = tokenizer(query, { text_pair: passage, truncation: true });  // for the tensor shapes
  const { Tensor } = await import('@huggingface/transformers');
  const idsT = new Tensor('int64', BigInt64Array.from(input_ids.map(BigInt)), [1, input_ids.length]);
  const maskT = new Tensor('int64', BigInt64Array.from(input_ids.map(() => 1n)), [1, input_ids.length]);
  const out = await model({ input_ids: idsT, attention_mask: maskT });
  void inputs;
  return sigmoid(out.logits.tolist()[0][0]);
}

const results = { A: [], B: [], C: [] };
let n = 0;
for (const q of python) {
  for (const r of q.reranked) {
    const tokens = tokenizer(q.question, { text_pair: body[r.chunk_id] }).input_ids.dims[1];
    if (tokens <= MAX) continue;
    n += 1;
    for (const s of ['A', 'B', 'C']) {
      results[s].push(Math.abs((await score(q.question, body[r.chunk_id], s)) - r.logit));
    }
  }
}
console.log(`${n} pairs over ${MAX} tokens; specials=${SPECIALS}`);
for (const s of ['A', 'B', 'C']) {
  const xs = results[s];
  console.log(`  ${s}: max |diff| ${Math.max(...xs).toFixed(6)}   mean ${(xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(6)}   pairs within 1e-4: ${xs.filter((x) => x < 1e-4).length}/${xs.length}`);
}
