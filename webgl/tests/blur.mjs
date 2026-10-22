#!/usr/bin/env node
// V2 motion blur: engine.synthBlur vs the library's recordSynthBlur (vkhdr.cpp --blur, Rgba8) on a few pairs, V2 Performance and
// Quality, flowScale 2, three windows (sample counts, both weightings, a window ending on curr).
//   node webgl/tests/blur.mjs [--pairs a,b] [--modes v2p,v2q]
// Gate: out MAE <= 0.5 levels and % px > 4 levels off <= 1 (the cross-backend tolerances). Exit 0 PASS, 1 FAIL, 2 setup error.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { openEngine } from './server.mjs';
import { repo, work, here } from './paths.mjs';

const argv = process.argv.slice(2);
const opt = { pairs: ['panel', 'objects', 'zoom', 'battlefield'], modes: ['v2p', 'v2q'] };
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--pairs') opt.pairs = argv[++i].split(',');
  else if (argv[i] === '--modes') opt.modes = argv[++i].split(',');
  else { console.error('blur: unknown argument ' + argv[i]); process.exit(2); }
}
const WINDOWS = [
  { tag: 'w25_75_n4_gauss', t0: 0.25, t1: 0.75, n: 4, weights: 'gaussian' },
  { tag: 'w00_50_n3_even', t0: 0.0, t1: 0.5, n: 3, weights: 'even' },
  { tag: 'w50_100_n5_gauss', t0: 0.5, t1: 1.0, n: 5, weights: 'gaussian' },
];
const SET = { v2p: ['algo=v2,v2Mode=performance', { motionVersion: 2, v2Mode: 'performance' }], v2q: ['algo=v2,v2Mode=quality', { motionVersion: 2, v2Mode: 'quality' }] };
const vkhdr = path.join(work, 'bin', 'vkhdr');
if (!fs.existsSync(vkhdr)) {
  fs.mkdirSync(path.dirname(vkhdr), { recursive: true });
  const r = spawnSync('g++', ['-O2', '-std=c++17', '-I', path.join(repo, 'gpu'), path.join(here, 'vkhdr.cpp'),
    path.join(repo, 'gpu', 'build-webgl', 'libframegen.a'), '-lvulkan', '-o', vkhdr], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(2);
}
const man = JSON.parse(fs.readFileSync(path.join(work, 'pairs', 'manifest.json'), 'utf8'));
const pairs = man.pairs.filter((p) => opt.pairs.includes(p.name));
for (const p of pairs) for (const mode of opt.modes) for (const w of WINDOWS) {
  const d = path.join(work, 'vkblur', mode + '_fs2', p.name);
  const f = path.join(d, w.tag + '.rgba');
  if (fs.existsSync(f)) continue;
  fs.mkdirSync(d, { recursive: true });
  const r = spawnSync(vkhdr, ['--set', SET[mode][0] + ',flowScale=2', '--format', 'rgba8', '--blur', String(w.n), '--t0', String(w.t0), '--t', String(w.t1),
    '--weights', w.weights === 'even' ? 'even' : 'gauss', String(p.width), String(p.height), path.join(work, 'pairs', p.name, 'a.rgba'), path.join(work, 'pairs', p.name, 'c.rgba'), f], { encoding: 'utf8' });
  if (r.status !== 0) { console.error(`blur: vkhdr failed: ${r.stderr}`); process.exit(2); }
}
const E = await openEngine({ settings: { debug: true } });
if (E.init.fatal) { console.error('blur: ' + E.init.fatal); process.exit(2); }
console.log(`browser : Chromium ${E.version}, renderer ${E.init.info.renderer}`);
let allOk = true;
const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : String(v));
for (const mode of opt.modes) for (const p of pairs) {
  const windows = WINDOWS.map((w) => ({ ...w, ref: `vkblur/${mode}_fs2/${p.name}/${w.tag}.rgba` }));
  const r = await E.page.evaluate((j) => window.__x.runBlur(j), { pair: p.name, W: p.width, H: p.height, settings: { ...SET[mode][1], flowScale: 2 }, windows });
  if (r.error) { console.log(`${mode} ${p.name}: ENGINE ERROR ${r.error.error}\n${r.error.stack}`); allOk = false; continue; }
  for (const w of WINDOWS) {
    const T = r.w[w.tag];
    const ok = T.ok && T.cmp && T.cmp.mae <= 0.5 && T.cmp.pct4 <= 1.0 && !r.glError;
    if (!ok) allOk = false;
    console.log(`${mode}_fs2 ${p.name.padEnd(12)} ${w.tag.padEnd(18)} ${ok ? 'PASS' : 'FAIL'}  MAE ${f3(T.cmp && T.cmp.mae)} max ${T.cmp && T.cmp.max} %>4 ${f3(T.cmp && T.cmp.pct4)} PSNR ${f3(T.cmp && T.cmp.psnr)}`);
  }
}
await E.browser.close();
console.log(`RESULT: ${allOk ? 'PASS' : 'FAIL'}`);
process.exit(allOk ? 0 : 1);
