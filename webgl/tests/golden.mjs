#!/usr/bin/env node
// Golden harness, browser side: drives the UNCHANGED extension/engine/framegen.js in headless Chromium
// (SwiftShader WebGL2) on the test pairs written by ref_gen.py and compares against proto/interp.py.
//
//   node golden.mjs [--engine FILE] [--shaders DIR] [--ref DIR] [--pairs a,b] [--ts 0.5,0.25]
//                   [--gate-ts 0.5,0.25] [--verbose] [--dump] [--no-debug]
//
// Exit: 0 = PASS (D11), 1 = compare failure, 2 = setup/environment/engine-load error.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch } from './browser.mjs';
import { pairStages, tStages, compareStage, rgb } from './compare.mjs';
import { encodePng } from './png.mjs';

import { here, work, engineFile, shaderDir } from './paths.mjs';

// ------------------------------------------------------------------ args
const argv = process.argv.slice(2);
const opt = { engine: engineFile, shaders: shaderDir,
  ref: path.join(work, 'ref'), pairs: null, ts: null, gateTs: '0.5,0.25', verbose: false, dump: false, debug: true };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], nx = () => argv[++i];
  if (a === '--engine') opt.engine = path.resolve(nx());
  else if (a === '--shaders') opt.shaders = path.resolve(nx());
  else if (a === '--ref') opt.ref = path.resolve(nx());
  else if (a === '--pairs') opt.pairs = nx().split(',');
  else if (a === '--ts') opt.ts = nx().split(',').map(Number);
  else if (a === '--gate-ts') opt.gateTs = nx();
  else if (a === '--verbose' || a === '-v') opt.verbose = true;
  else if (a === '--dump') opt.dump = true;
  else if (a === '--no-debug') opt.debug = false;
  else if (a === '-h' || a === '--help') { console.log(fs.readFileSync(fileURLToPath(import.meta.url), 'utf8').split('\n').slice(1, 9).map((l) => l.replace(/^\/\/ ?/, '')).join('\n')); process.exit(0); }
  else { console.error('unknown argument: ' + a); process.exit(2); }
}
const gateTs = new Set(opt.gateTs.split(',').filter(Boolean).map(Number));
const rel = (p) => { const r = path.relative(process.cwd(), p); return r.startsWith('..') ? p : r; };
const die = (msg, code = 2) => { console.error('golden: ' + msg); process.exit(code); };

// ------------------------------------------------------------------ reference data
const manifestPath = path.join(opt.ref, 'manifest.json');
if (!fs.existsSync(manifestPath)) die(`no reference manifest at ${manifestPath}\n  run: python3 ${path.join(here, 'ref_gen.py')}  (or tools/golden/run.sh)`);
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const W = manifest.width, H = manifest.height;

function readArr(entry) {
  const buf = fs.readFileSync(path.join(opt.ref, entry.file));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const data = entry.dtype === 'f4' ? new Float32Array(ab) : entry.dtype === 'u1' ? new Uint8Array(ab) : null;
  if (!data) die('unsupported dtype ' + entry.dtype);
  const [h, w, c] = entry.shape;
  if (data.length !== h * w * c) die(`${entry.file}: size ${data.length} != ${h}*${w}*${c}`);
  return { w, h, c, data };
}
const b64 = (arr) => Buffer.from(arr.data.buffer, arr.data.byteOffset, arr.data.byteLength).toString('base64');
function decode(p) { // page payload {type,b64,width,height,channels} -> engine array
  const buf = Buffer.from(p.b64, 'base64');
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const ctor = { f4: Float32Array, u1: Uint8Array, u2: Uint16Array, u4: Uint32Array }[p.type];
  return { ...p, data: new ctor(ab) };
}

const pairsSel = manifest.pairs.filter((p) => !opt.pairs || opt.pairs.includes(p.name));
if (!pairsSel.length) die('no pairs selected');
const tsAll = (opt.ts || manifest.ts);
for (const t of tsAll) if (!manifest.ts.includes(t)) die(`t=${t} not in reference (has ${manifest.ts.join(',')}); regenerate with ref_gen.py --ts`);

// ------------------------------------------------------------------ browser
if (!fs.existsSync(opt.engine)) die(`engine not found: ${opt.engine}`);
const { browser, playwright, executablePath } = await launch().catch((e) => die(e.message));
const consoleMsgs = new Map();
const noteConsole = (kind, text) => { if (/GPU stall due to ReadPixels/.test(text)) return; // expected: debugRead/stats are readbacks
   const k = kind + ': ' + text.slice(0, 300); consoleMsgs.set(k, (consoleMsgs.get(k) || 0) + 1); };
const flushConsole = () => { if (consoleMsgs.size) { console.log('\npage console (unique, count):'); let n = 0; for (const [k, v] of consoleMsgs) { if (n++ >= 15) { console.log('  ...'); break; } console.log(`  [x${v}] ${k}`); } } };
const bail = async (msg, code = 2) => { flushConsole(); await browser.close().catch(() => {}); die(msg, code); };

const page = await browser.newPage();
page.on('console', (m) => { if (['warning', 'error'].includes(m.type())) noteConsole(m.type(), m.text()); });
page.on('pageerror', (e) => noteConsole('pageerror', String(e.stack || e)));
await page.goto('about:blank');
await page.addScriptTag({ content: fs.readFileSync(opt.engine, 'utf8') }); // unchanged
const shaderFiles = await page.evaluate(() => (typeof __fg === 'object' && __fg ? Array.from(__fg.shaderFiles || []) : null));
if (shaderFiles === null) await bail(`${opt.engine} did not define globalThis.__fg`);
const sources = {}, missing = [];
for (const f of shaderFiles) {
  const p = path.join(opt.shaders, f);
  if (fs.existsSync(p)) sources[f] = fs.readFileSync(p, 'utf8'); else missing.push(f);
}
if (missing.length) await bail(`__fg.shaderFiles lists files missing under ${opt.shaders}:\n  ${missing.join('\n  ')}`);
await page.addScriptTag({ content: fs.readFileSync(path.join(here, 'page.js'), 'utf8') });

const settings = { flowScale: manifest.flow_scale, debug: opt.debug };
const init = await page.evaluate((a) => window.__golden.init(a), { sources, settings });
const inf = init.info || {};
console.log(`browser   : Chromium ${browser.version()} (${executablePath || 'playwright default'}), playwright-core from ${playwright}`);
console.log(`renderer  : ${inf.renderer}   ${inf.version}`);
console.log(`extensions: ${Object.entries(inf.extensions || {}).map(([k, v]) => `${k}=${v ? 'yes' : 'NO'}`).join('  ')}  MAX_DRAW_BUFFERS=${inf.maxDrawBuffers}`);
console.log(`engine    : ${rel(opt.engine)} (framegen version ${inf.fgVersion}, ${shaderFiles.length} shader files from ${rel(opt.shaders)}), settings ${JSON.stringify(settings)}`);
console.log(`reference : ${manifest.reference.dir} (main@${manifest.reference.pinned_commit}, interp.py sha ${manifest.reference.interp_sha256.slice(0, 12)}), ${W}x${H}, flowScale ${manifest.flow_scale}`);
if (init.fatal) await bail('engine init failed: ' + init.fatal);
if (inf.extensions && Object.values(inf.extensions).some((v) => !v)) await bail('required WebGL2 extension missing: ' + Object.entries(inf.extensions).filter(([, v]) => !v).map(([k]) => k).join(', '));
if (inf.support && !inf.support.ok) await bail('__fg.checkSupport reports missing: ' + inf.support.missing.join(', '));

// ------------------------------------------------------------------ run + compare
const f3 = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(3)) : String(v));
const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : String(v));
const pad = (s, n) => String(s).padEnd(n);
const lastFail = { any: false };
const results = [];
const dumpDir = path.join(work, 'dump');
if (opt.dump) fs.mkdirSync(dumpDir, { recursive: true });

function stageRows(specs, getEng, getRef, W_, H_) {
  const rows = [];
  for (const sp of specs) {
    const r = getRef(sp.name), e = getEng(sp.name);
    if (!r) { rows.push({ sp, res: { status: 'ERROR', error: 'no reference array (harness bug)' } }); continue; }
    if (!e) { rows.push({ sp, res: { status: 'ERROR', error: 'not read' } }); continue; }
    const res = compareStage(sp, e.error ? e : decode(e), r);
    if (res.status === 'CMP') res.diverged = !sp.ok(res.m, W_, H_);
    else res.diverged = true;
    rows.push({ sp, res });
  }
  return rows;
}

function printRows(rows, indent = '  ') {
  console.log(`${indent}${pad('stage', 12)} ${pad('mean', 9)} ${pad('max', 9)} ${pad('%differ', 8)} ${pad('%>4', 6)} status   worst @(x,y) engine vs ref`);
  let first = true;
  for (const { sp, res } of rows) {
    if (res.status !== 'CMP') { console.log(`${indent}${pad(sp.stage, 12)} ${res.status}: ${res.error}${first ? '   <-- first divergence' : ''}`); first = false; continue; }
    const m = res.m;
    const mark = res.diverged ? (first ? 'DIVERGED  <-- first divergence' : 'DIVERGED') : 'ok';
    if (res.diverged) first = false;
    const w = m.worst;
    console.log(`${indent}${pad(sp.stage, 12)} ${pad(f3(m.mean), 9)} ${pad(f3(m.max), 9)} ${pad(f2(m.pctDiffer), 8)} ${pad(m.pct4 === undefined ? '' : f2(m.pct4), 6)} ${pad(mark, 8)} ${m.max > 0 ? `(${w.x},${w.y}) ${JSON.stringify(w.eng)} vs ${JSON.stringify(w.ref)}` : ''}`);
  }
}

for (const pair of pairsSel) {
  const pairArrays = {};
  for (const [k, v] of Object.entries(pair.arrays)) pairArrays[k] = readArr(v);
  const pairStageSpecs = pairStages(pair.levels);
  const pairNames = [...new Set(pairStageSpecs.map((s) => s.name))];
  const tNames = [...new Set(tsAll.flatMap((t) => tStages(t).map((s) => s.name)))];
  const run = await page.evaluate((a) => window.__golden.runPair(a), {
    w: W, h: H, a: b64(pairArrays['frame.prev']), c: b64(pairArrays['frame.curr']), ts: tsAll, pairNames, tNames,
  });
  console.log(`\n=== pair ${pair.name}: ${pair.desc}`);
  const callErrs = Object.entries(run.calls).filter(([, v]) => v && v.error);
  const R = { pair: pair.name, ts: {}, pass: true, gated: true };
  results.push(R);
  if (callErrs.length) {
    for (const [k, v] of callErrs) console.log(`  ENGINE ERROR in ${k}: ${v.error}\n${(v.stack || '').split('\n').map((l) => '      ' + l).join('\n')}`);
    R.pass = false; R.error = callErrs[0][1].error;
    continue;
  }
  console.log(`  calls: ${Object.entries(run.calls).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('  ')}${run.passCounts ? '  passCounts=' + JSON.stringify(run.passCounts) : ''}${run.glError ? '  gl.getError()=' + run.glError : ''}`);
  // layout
  const lv = run.info && run.info.levels;
  const layoutOk = lv && lv.length === pair.levels.length && lv.every((l, i) => ['w', 'h', 'bw', 'bh'].every((k) => l[k] === pair.levels[i][k]));
  console.log(`  levels: engine ${lv ? lv.map((l) => `${l.w}x${l.h}`).join(' ') : '?'} vs reference ${pair.levels.map((l) => `${l.w}x${l.h}`).join(' ')}  ${layoutOk ? 'ok' : 'MISMATCH (engine.info.levels vs interp pyramid)'}`);

  // pair-level stages
  const prows = stageRows(pairStageSpecs, (n) => run.est[n], (n) => pairArrays[n], W, H);
  const pdiv = prows.find((r) => r.res.diverged);
  R.pairFirst = pdiv ? pdiv.sp.stage : null;

  for (const t of tsAll) {
    const rt = run.t[t];
    const refT = pair.t[String(t)];
    const gating = gateTs.has(t);
    const T = { t, gating, ok: true, reasons: [] };
    R.ts[t] = T;
    if (rt.calls.synth && rt.calls.synth.error) { T.ok = false; T.reasons.push('synth threw: ' + rt.calls.synth.error); console.log(`  t=${t}: ENGINE ERROR in synth: ${rt.calls.synth.error}\n${(rt.calls.synth.stack || '').split('\n').map((l) => '      ' + l).join('\n')}`); continue; }
    if (rt.calls.synth === false) { T.ok = false; T.reasons.push('synth returned false (no estimate?)'); continue; }
    const refArr = {};
    for (const [k, v] of Object.entries(refT.arrays)) refArr[k] = readArr(v);
    const trows = stageRows(tStages(t), (n) => rt.stages[n], (n) => refArr[n], W, H);
    const outRow = trows.find((r) => r.sp.stage === 'out');
    T.out = outRow.res.status === 'CMP' ? outRow.res.m : null;
    if (!T.out) { T.ok = false; T.reasons.push('out: ' + outRow.res.error); }
    else if (!(T.out.mean <= 1)) { T.ok = false; T.reasons.push(`MAE ${f3(T.out.mean)} > 1 level`); }
    // decisions
    const st = rt.stats || {};
    const dec = { fallback: st.fallback, cut: st.cut, flick: st.flick };
    const decOk = st.error === undefined && dec.fallback === refT.fallback && dec.cut === refT.cut && (refT.cut || dec.flick === refT.flick);
    if (!decOk) { T.ok = false; T.reasons.push(`decisions differ: engine ${JSON.stringify(dec)} vs reference ${JSON.stringify({ fallback: refT.fallback, cut: refT.cut, flick: refT.flick })}${st.error ? ' stats(): ' + st.error : ''}`); }
    // fbo blit == out exactly
    let blit = null;
    if (rt.fbo && rt.stages.out && !rt.stages.out.error) {
      const fo = decode(rt.fbo), eo = decode(rt.stages.out);
      blit = rgb({ w: W, h: H, c: 4, data: fo.data }, { w: W, h: H, c: 4, data: eo.data });
      if (blit.max !== 0) { T.ok = false; T.reasons.push(`synth(t, fbo, {0,0,W,H}) blit differs from debugRead('out'): max ${blit.max}, mean ${f3(blit.mean)} (worst @(${blit.worst.x},${blit.worst.y}) fbo ${JSON.stringify(blit.worst.eng)} vs out ${JSON.stringify(blit.worst.ref)}) - y-flip/blit bug? (LAYOUT 5.1)`); }
    }
    T.dec = dec; T.blit = blit;
    const nearThr = [];
    if (refT.mean_sad !== undefined && Math.abs(refT.mean_sad - manifest.params.CUT_SAD) < 0.2) nearThr.push('meanSad within 0.2 of CUT_SAD');
    if (Math.abs(refT.untrusted_share - manifest.params.GLOBAL_FRAC) < 0.003) nearThr.push('untrusted share within 0.003 of GLOBAL_FRAC');
    T.stats = st;
    const oMae = T.out ? `MAE ${f3(T.out.mean)}  max ${f3(T.out.max)}  %px>4 ${f2(T.out.pct4)}` : 'out ?';
    console.log(`  t=${pad(t, 5)} ${T.ok ? 'PASS' : 'FAIL'}${gating ? '' : ' (info, not gating)'}  ${oMae}  | engine meanSad ${f3(st.meanSad)} vs ref ${f3(refT.mean_sad)}; fallback engine ${JSON.stringify(dec.fallback)} vs ref ${JSON.stringify(refT.fallback)}; cut ${dec.cut}/${refT.cut}; untrusted share ${st.untrustedShare == null ? '?' : f3(st.untrustedShare)} vs ${f3(refT.untrusted_share)}${nearThr.length ? '  [near threshold: ' + nearThr.join('; ') + ']' : ''}`);
    for (const r of T.reasons) console.log(`      - ${r}`);
    const failed = !T.ok && gating;
    if (failed) R.pass = false;
    // stage tables
    const tdiv = trows.find((r) => r.res.diverged);
    T.first = pdiv ? pdiv.sp.stage : (tdiv ? tdiv.sp.stage : null);
    const nDiv = trows.filter((r) => r.res.diverged).length + prows.filter((r) => r.res.diverged).length;
    const divNames = [...prows, ...trows].filter((r) => r.res.diverged).map((r) => r.sp.stage);
    console.log(`      stages: ${prows.length + trows.length - nDiv}/${prows.length + trows.length} within informational tolerances${nDiv ? ' (diverged: ' + divNames.join(', ') + '; --verbose for the tables)' : ''}`);
    if (opt.verbose || !T.ok || (!layoutOk)) {
      if (!R.printedPairRows) { console.log('  -- estimate stages (pipeline order, earliest first):'); printRows(prows, '    '); R.printedPairRows = true; }
      console.log(`  -- synth stages at t=${t}:`); printRows(trows, '    ');
      console.log(`  first divergent stage (earliest in pipeline): ${T.first || 'none (all stages within informational tolerances)'}`);
    }
    if (opt.dump && rt.stages.out && !rt.stages.out.error) {
      const eo = decode(rt.stages.out), ro = refArr.out;
      const strip = new Uint8Array(W * 3 * H * 4);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let k = 0; k < 4; k++) {
        const s = (y * W + x) * 4 + k, d = (y * W * 3 + x) * 4 + k;
        const ev = eo.data[s], rv = ro.data[s];
        strip[d] = ev; strip[d + W * 4] = rv; strip[d + 2 * W * 4] = k === 3 ? 255 : Math.min(255, Math.abs(ev - rv) * 8);
      }
      const fn = path.join(dumpDir, `${pair.name}_t${t}.png`);
      fs.writeFileSync(fn, encodePng(W * 3, H, strip));
      console.log(`      dump: ${rel(fn)}  (engine | reference | |diff|x8)`);
    }
  }
  if (opt.verbose && R.pairFirst) console.log(`  estimate-stage first divergence: ${R.pairFirst}`);
}

// ------------------------------------------------------------------ summary
flushConsole();
await browser.close();
console.log('\n=== summary (PASS = every gated t has MAE<=1 level AND identical scene-cut/fallback decisions, plus fbo blit exact)');
console.log(`${pad('pair', 10)} ${tsAll.map((t) => pad('t=' + t, 22)).join(' ')} first-divergent-stage`);
for (const R of results) {
  const cells = tsAll.map((t) => { const T = R.ts[t]; return pad(!T ? 'ERROR' : `${T.ok ? 'pass' : 'FAIL'}${T.out ? ' mae ' + f3(T.out.mean) : ''}${T.gating ? '' : '*'}`, 22); });
  const firstDiv = R.error ? 'engine threw' : (R.pairFirst || Object.values(R.ts).map((T) => T.first).find(Boolean) || 'none (decisions / fbo blit)');
  console.log(`${pad(R.pair, 10)} ${cells.join(' ')} ${R.pass ? '' : firstDiv}`);
}
const pass = results.every((R) => R.pass);
console.log(`(* = informational, not gating)\nRESULT: ${pass ? 'PASS' : 'FAIL'}`);
process.exit(pass ? 0 : 1);
