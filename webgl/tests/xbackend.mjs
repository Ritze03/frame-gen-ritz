#!/usr/bin/env node
// Cross-backend reference test: the WebGL2 engine (headless Chromium, SwiftShader) vs the Vulkan library (vkref.mjs output) on the same
// pairs (pairs.py) for V1, V2 Performance and V2 Quality at each pair's flow scales and t in {0.5, 0.25}.
//   node webgl/tests/xbackend.mjs [--modes v1,v2p,v2q] [--pairs a,b] [--fs 1,2,4] [--static 0|1] [--dump] [--json FILE] [--gpu]
// Gate per pair x mode x flowScale x t (README "Tolerances"): identical whole-frame decisions (scene cut; flick when no cut), and
//   no fallback: out MAE <= TOL[mode].mae levels and % pixels with a channel > 4 levels off <= TOL[mode].pct4;
//   fallback: out bit-identical (both show the nearer real frame).
// Plus the switch semantics (motionVersion drops the estimate, v2Mode keeps it). Exit 0 PASS, 1 FAIL, 2 setup error.
import fs from 'node:fs';
import path from 'node:path';
import { openEngine } from './server.mjs';
import { work } from './paths.mjs';
import { encodePng } from './png.mjs';

export const TOL = {
  v1: { mae: 0.5, pct4: 1.0 },
  v2p: { mae: 0.5, pct4: 1.0 },
  v2q: { mae: 0.5, pct4: 1.0 },
};
const argv = process.argv.slice(2);
const opt = { modes: ['v1', 'v2p', 'v2q'], pairs: null, fs: null, staticMask: 1, dump: false, json: null, gpu: false };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], nx = () => argv[++i];
  if (a === '--modes') opt.modes = nx().split(',');
  else if (a === '--pairs') opt.pairs = nx().split(',');
  else if (a === '--fs') opt.fs = nx().split(',').map(Number);
  else if (a === '--static') opt.staticMask = Number(nx());
  else if (a === '--dump') opt.dump = true;
  else if (a === '--json') opt.json = nx();
  else if (a === '--gpu') opt.gpu = true;
  else { console.error('xbackend: unknown argument ' + a); process.exit(2); }
}
const man = JSON.parse(fs.readFileSync(path.join(work, 'pairs', 'manifest.json'), 'utf8'));
const E = await openEngine({ gpu: opt.gpu, settings: { debug: true } });
console.log(`browser : Chromium ${E.version}, renderer ${E.init.info && E.init.info.renderer}, MAX_TEXTURE_SIZE ${E.init.info && E.init.info.maxTex}`);
if (E.init.fatal) { console.error('xbackend: engine init failed: ' + E.init.fatal); await E.browser.close(); process.exit(2); }

const SET = {
  v1: { motionVersion: 1 },
  v2p: { motionVersion: 2, v2Mode: 'performance' },
  v2q: { motionVersion: 2, v2Mode: 'quality' },
};
const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : String(v));
const pad = (s, n) => String(s).padEnd(n);
const rows = [];
let allOk = true;
for (const mode of opt.modes) {
  for (const p of man.pairs) {
    if (opt.pairs && !opt.pairs.includes(p.name)) continue;
    for (const S of p.flow_scales) {
      if (opt.fs && !opt.fs.includes(S)) continue;
      const cfg = `${mode}${mode !== 'v1' && !opt.staticMask ? '_s0' : ''}_fs${S}`;
      const settings = { ...SET[mode], flowScale: S, v2StaticMask: !!opt.staticMask };
      const r = await E.page.evaluate((j) => window.__x.run(j), { pair: p.name, W: p.width, H: p.height, settings, ts: man.ts, ref: `vkref/${cfg}/${p.name}`, mode, dump: opt.dump, dumpName: `${cfg}_${p.name}` });
      if (r.error) { console.log(`${cfg} ${p.name}: ENGINE ERROR ${r.error.error}\n${r.error.stack}`); allOk = false; rows.push({ cfg, pair: p.name, t: '-', ok: false, err: r.error.error }); continue; }
      for (const t of man.ts) {
        const T = r.t[t];
        if (!T || !T.synth) { console.log(`${cfg} ${p.name} t=${t}: synth returned ${T && T.synth}`); allOk = false; rows.push({ cfg, pair: p.name, t, ok: false }); continue; }
        if (T.noRef) { console.log(`${cfg} ${p.name} t=${t}: no Vulkan reference (run vkref.mjs)`); allOk = false; continue; }
        const [libUntr, libSad256, libBlocks] = T.libStats;
        const W = p.width, H = p.height;
        const libMean = libSad256 / 256 / libBlocks, libCut = libMean > 30, libFlick = libUntr > 0.15 * W * H;
        const es = T.stats;
        const decOk = es.cut === libCut && (libCut || es.flick === libFlick);
        const fallback = libCut || libFlick;
        const tol = TOL[mode];
        const imgOk = fallback ? T.cmp.max === 0 : (T.cmp.mae <= tol.mae && T.cmp.pct4 <= tol.pct4);
        const near = Math.abs(libMean - 30) < 0.25 || Math.abs(libUntr / (W * H) - 0.15) < 0.003;
        const ok = decOk && imgOk && !r.glError;
        if (!ok) allOk = false;
        const st = T.stages || {};
        rows.push({ cfg, mode, S, pair: p.name, W, H, t, ok, decOk, fallback: fallback ? (libCut ? 'cut' : 'flick') : null, cmp: T.cmp, stages: st, near,
          eng: { meanSad: es.meanSad, untrusted: es.untrusted }, lib: { meanSad: libMean, untrusted: libUntr }, ms: r.ms });
        const stg = [st.blockField && `block EPE ${f3(st.blockField.mean)}/${f3(st.blockField.max)} (${f3(st.blockField.pctDiffer)}%)`,
          st.pixField && `pix EPE ${f3(st.pixField.mean)}/${f3(st.pixField.max)} (${f3(st.pixField.pctDiffer)}%)`,
          st.coverage && `cov ${f3(st.coverage.meanP)}/${f3(st.coverage.meanC)}/${f3(st.coverage.meanZ)}`].filter(Boolean).join('  ');
        console.log(`${pad(cfg, 9)} ${pad(p.name, 12)} ${pad(W + 'x' + H, 10)} t=${pad(t, 5)} ${ok ? 'PASS' : 'FAIL'}  MAE ${f3(T.cmp.mae)} max ${pad(T.cmp.max, 3)} %>4 ${f3(T.cmp.pct4)} PSNR ${f3(T.cmp.psnr)}` +
          `  dec ${decOk ? 'same' : 'DIFF'}${fallback ? ' (' + (libCut ? 'cut' : 'flick') + ')' : ''}${near ? ' [near threshold]' : ''}` +
          `  sad ${f3(es.meanSad)}/${f3(libMean)} untr ${es.untrusted}/${libUntr}  ${stg}  ${r.ms ? Math.round(r.ms) + ' ms' : ''}`);
        if (opt.dump && E.dumps.has(`${cfg}_${p.name}_t${t}`)) {
          const eng = new Uint8Array(E.dumps.get(`${cfg}_${p.name}_t${t}`));
          const lib = new Uint8Array(fs.readFileSync(path.join(work, 'vkref', cfg, p.name, `t${t}.rgba`)));
          const strip = new Uint8Array(W * 3 * H * 4);
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let k = 0; k < 4; k++) {
            const s = (y * W + x) * 4 + k, d = (y * W * 3 + x) * 4 + k;
            strip[d] = eng[s]; strip[d + W * 4] = lib[s]; strip[d + 2 * W * 4] = k === 3 ? 255 : Math.min(255, Math.abs(eng[s] - lib[s]) * 8);
          }
          const dd = path.join(work, 'dump'); fs.mkdirSync(dd, { recursive: true });
          fs.writeFileSync(path.join(dd, `${cfg}_${p.name}_t${t}.png`), encodePng(W * 3, H, strip));
        }
      }
    }
  }
}
// switch semantics
const sp = man.pairs.find((p) => p.name === 'panel') || man.pairs[0];
await E.page.evaluate((a) => window.__x.setSettings(a), { flowScale: 2 });
const sw = await E.page.evaluate((a) => window.__x.switchTest(a), { pair: sp.name, W: sp.width, H: sp.height, t: 0.25 });
const want = { estQ: true, synthQ: true, 'synthP-kept': true, estP: true, synthP: true, 'synthQ-kept': true, 'synth1-before-estimate': false, est1: true, synth1: true, 'synth2-before-estimate': false, est2: true, synth2: true };
const swOk = !sw.error && sw.steps.every(([k, v]) => want[k] === v) && sw.diff.pKeptVsOwn === 0 && sw.diff.qLazyVsOwn === 0 && sw.diff.qRoundTrip === 0 && !sw.glError;
console.log(`\nswitch test on ${sp.name}: ${swOk ? 'PASS' : 'FAIL'} steps ${JSON.stringify(sw.steps)} diffs ${JSON.stringify(sw.diff)}${sw.error ? ' ERROR ' + sw.error.error : ''}`);
if (!swOk) allOk = false;
if (E.consoleMsgs.size) { console.log('\npage console:'); for (const [k, v] of E.consoleMsgs) console.log(`  [x${v}] ${k}`); }
await E.browser.close();

console.log('\n=== summary per config (worst over pairs and t)');
const byCfg = new Map();
for (const r of rows) { if (!byCfg.has(r.cfg)) byCfg.set(r.cfg, []); byCfg.get(r.cfg).push(r); }
for (const [cfg, rs] of byCfg) {
  const ok = rs.filter((r) => r.ok).length, cm = rs.filter((r) => r.cmp);
  const worst = cm.reduce((a, r) => (r.cmp.mae > (a ? a.cmp.mae : -1) ? r : a), null);
  const meanMae = cm.reduce((s, r) => s + r.cmp.mae, 0) / Math.max(1, cm.length);
  console.log(`${pad(cfg, 9)} ${ok}/${rs.length} pass  mean MAE ${f3(meanMae)}  worst MAE ${worst ? f3(worst.cmp.mae) + ' (' + worst.pair + ' t=' + worst.t + ')' : '-'}  worst %>4 ${f3(Math.max(...cm.map((r) => r.cmp.pct4)))}  decisions ${rs.every((r) => r.decOk) ? 'all identical' : 'DIFFER'}`);
}
if (opt.json) fs.writeFileSync(opt.json, JSON.stringify({ rows, switchTest: sw }, null, 1));
console.log(`RESULT: ${allOk ? 'PASS' : 'FAIL'}`);
process.exit(allOk ? 0 : 1);
