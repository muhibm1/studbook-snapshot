// Fetches the certificate chain the database host presents and saves its root as PEM so the
// gate can verify against it. Trust-on-first-use: the root is taken from the connection itself,
// so cross-check its subject against the CA certificate downloadable from the Supabase dashboard
// (Project Settings -> Database -> SSL) before treating it as authoritative. Prints subject and
// issuer names only; never the connection string.
import tls from 'node:tls';
import { writeFileSync } from 'node:fs';

const text = process.env.STUDBOOK_DATABASE_URL ?? '';
const kv = Object.fromEntries([...text.matchAll(/(\w+)\s*=\s*(?:'((?:\\.|[^'])*)'|(\S+))/g)].map((m) => [m[1], m[2] ?? m[3]]));
const host = kv.host, port = Number(kv.port || 5432);
if (!host) { console.error('no host in STUDBOOK_DATABASE_URL'); process.exit(2); }

// Postgres speaks SSLRequest before TLS: send the 8-byte request, expect 'S', then handshake.
import net from 'node:net';
const sock = net.connect(port, host, () => {
  sock.write(Buffer.from([0, 0, 0, 8, 0x04, 0xd2, 0x16, 0x2f]));
});
sock.once('data', (b) => {
  if (b.toString() !== 'S') { console.error('server refused SSL'); process.exit(1); }
  const t = tls.connect({ socket: sock, servername: host, rejectUnauthorized: false }, () => {
    let cert = t.getPeerCertificate(true), depth = 0, root = null;
    const seen = new Set();
    while (cert && !seen.has(cert.fingerprint256)) {
      seen.add(cert.fingerprint256);
      console.log(`[${depth}] subject: ${cert.subject?.CN ?? cert.subject?.O}  |  issuer: ${cert.issuer?.CN ?? cert.issuer?.O}  |  valid to: ${cert.valid_to}`);
      root = cert; depth += 1;
      cert = cert.issuerCertificate;
    }
    const pem = `-----BEGIN CERTIFICATE-----\n${root.raw.toString('base64').match(/.{1,64}/g).join('\n')}\n-----END CERTIFICATE-----\n`;
    writeFileSync(new URL('./supabase-ca.pem', import.meta.url), pem);
    console.log(`saved root [${depth - 1}] to gate/supabase-ca.pem (sha256 ${root.fingerprint256})`);
    t.end();
  });
});
