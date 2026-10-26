// Headless Chromium with the engine loaded on a local HTTP server (127.0.0.1, random port; large frames do not fit through the
// DevTools protocol): /w/<path> serves files of $FG_WORK, POST /dump/<name> stores a body (engine output) for the caller.
// Shared by xbackend.mjs, hdr.mjs, blur.mjs, bench.mjs and learned.mjs.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { launch } from './browser.mjs';
import { here, work, engineFile, shaderDir } from './paths.mjs';

function serve(dumps) {
  const srv = http.createServer((req, res) => {
    const u = new URL(req.url, 'http://x');
    if (u.pathname === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>fg</title>'); return; }
    if (u.pathname.startsWith('/w/')) {
      const f = path.join(work, decodeURIComponent(u.pathname.slice(3)));
      if (!f.startsWith(work) || !fs.existsSync(f)) { res.writeHead(404); res.end('missing'); return; }
      res.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': fs.statSync(f).size });
      fs.createReadStream(f).pipe(res);
      return;
    }
    if (u.pathname.startsWith('/dump/') && req.method === 'POST') {
      const parts = [];
      req.on('data', (d) => parts.push(d));
      req.on('end', () => { dumps.set(decodeURIComponent(u.pathname.slice(6)), Buffer.concat(parts)); res.writeHead(200); res.end('ok'); });
      return;
    }
    res.writeHead(404); res.end('no route');
  });
  return new Promise((resolve) => srv.listen(0, '127.0.0.1', () => resolve(srv)));
}

export async function openEngine({ gpu = false, settings = {}, engine = engineFile, shaders = shaderDir } = {}) {
  const dumps = new Map();
  const srv = await serve(dumps);
  const { browser, executablePath } = await launch({ gpu });
  const close = browser.close.bind(browser);
  browser.close = async () => { await close(); srv.close(); };
  const consoleMsgs = new Map();
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (!['warning', 'error'].includes(m.type()) || /GPU stall due to ReadPixels/.test(m.text())) return;
    const k = m.type() + ': ' + m.text().slice(0, 300);
    consoleMsgs.set(k, (consoleMsgs.get(k) || 0) + 1);
  });
  page.on('pageerror', (e) => consoleMsgs.set('pageerror: ' + String(e.stack || e).slice(0, 300), 1));
  await page.goto(`http://127.0.0.1:${srv.address().port}/`);
  await page.addScriptTag({ content: fs.readFileSync(engine, 'utf8') });
  // the learned estimator's weights (webgl/nets/<net>.js, classic scripts next to framegen.js)
  const nets = path.join(path.dirname(engine), 'nets');
  if (fs.existsSync(nets)) for (const f of fs.readdirSync(nets).filter((x) => x.endsWith('.js')).sort()) await page.addScriptTag({ content: fs.readFileSync(path.join(nets, f), 'utf8') });
  const files = await page.evaluate(() => Array.from(globalThis.__fg.shaderFiles));
  const sources = {};
  for (const f of files) sources[f] = fs.readFileSync(path.join(shaders, f), 'utf8');
  await page.addScriptTag({ content: fs.readFileSync(path.join(here, 'xpage.js'), 'utf8') });
  const init = await page.evaluate((a) => window.__x.init(a), { sources, settings });
  return { browser, page, init, dumps, consoleMsgs, executablePath, version: browser.version() };
}
