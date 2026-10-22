// Headless Chromium with the engine loaded on http://fg.test/ (Playwright routes, no real server): /w/<path> serves files of
// $FG_WORK, POST /dump/<name> stores a body (engine output) for the caller. Shared by xbackend.mjs and bench.mjs.
import fs from 'node:fs';
import path from 'node:path';
import { launch } from './browser.mjs';
import { here, work, engineFile, shaderDir } from './paths.mjs';

export async function openEngine({ gpu = false, settings = {}, engine = engineFile, shaders = shaderDir } = {}) {
  const { browser, executablePath } = await launch({ gpu });
  const dumps = new Map();
  const consoleMsgs = new Map();
  const page = await browser.newPage();
  page.on('console', (m) => {
    if (!['warning', 'error'].includes(m.type()) || /GPU stall due to ReadPixels/.test(m.text())) return;
    const k = m.type() + ': ' + m.text().slice(0, 300);
    consoleMsgs.set(k, (consoleMsgs.get(k) || 0) + 1);
  });
  page.on('pageerror', (e) => consoleMsgs.set('pageerror: ' + String(e.stack || e).slice(0, 300), 1));
  await page.route('http://fg.test/**', async (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === '/') return route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>fg</title>' });
    if (u.pathname.startsWith('/w/')) {
      const f = path.join(work, decodeURIComponent(u.pathname.slice(3)));
      if (!f.startsWith(work) || !fs.existsSync(f)) return route.fulfill({ status: 404, body: 'missing' });
      return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: fs.readFileSync(f) });
    }
    if (u.pathname.startsWith('/dump/')) {
      dumps.set(decodeURIComponent(u.pathname.slice(6)), route.request().postDataBuffer());
      return route.fulfill({ status: 200, body: 'ok' });
    }
    return route.fulfill({ status: 404, body: 'no route' });
  });
  await page.goto('http://fg.test/');
  await page.addScriptTag({ content: fs.readFileSync(engine, 'utf8') });
  const files = await page.evaluate(() => Array.from(globalThis.__fg.shaderFiles));
  const sources = {};
  for (const f of files) sources[f] = fs.readFileSync(path.join(shaders, f), 'utf8');
  await page.addScriptTag({ content: fs.readFileSync(path.join(here, 'xpage.js'), 'utf8') });
  const init = await page.evaluate((a) => window.__x.init(a), { sources, settings });
  return { browser, page, init, dumps, consoleMsgs, executablePath, version: browser.version() };
}
