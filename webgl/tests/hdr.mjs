#!/usr/bin/env node
// HDR / float input test: the engine with inputFormat 'rgba16f' (transfer 'linear' = scRGB, 'encoded' = PQ or plain values) vs the
// Vulkan library in RgbaF16 (vkhdr.cpp) on the pairs of hdr_pairs.py, compared in levels of the perceptual encoding (PQ x 255 for
// linear, value x 255 for encoded), plus the identity encoding (8-bit values / 255 as floats) vs the library's Rgba8 output (vkref).
//   node webgl/tests/hdr.mjs [--modes v1,v2p,v2q] [--pairs a_lin,b_pq] [--fs 2] [--regen]
// Needs pairs.py and vkref.mjs output first. Builds $FG_WORK/bin/vkhdr (g++, against gpu/build-webgl/libframegen.a) if missing.
// A mode the library refuses in RgbaF16 (V2 before the V2 formats work) is reported as "no Vulkan reference" and only checked for
// finite output and against the identity encoding. Gate: identical decisions, MAE <= 0.5 levels, % px > 4 levels off <= 1.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { openEngine } from './server.mjs';
import { repo, work, here } from './paths.mjs';

const argv = process.argv.slice(2);
const opt = { modes: ['v1', 'v2p', 'v2q'], pairs: null, fs: [2], regen: false };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], nx = () => argv[++i];
  if (a === '--modes') opt.modes = nx().split(',');
  else if (a === '--pairs') opt.pairs = nx().split(',');
  else if (a === '--fs') opt.fs = nx().split(',').map(Number);
  else if (a === '--regen') opt.regen = true;
  else { console.error('hdr: unknown argument ' + a); process.exit(2); }
}
const TOL = { mae: 0.5, pct4: 1.0 };
const hdir = path.join(work, 'hdr');
if (opt.regen || !fs.existsSync(path.join(hdir, 'manifest.json'))) {
  const r = spawnSync('python3', [path.join(here, 'hdr_pairs.py')], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(2);
}
const vkhdr = path.join(work, 'bin', 'vkhdr');
if (!fs.existsSync(vkhdr)) {
  fs.mkdirSync(path.dirname(vkhdr), { recursive: true });
  const r = spawnSync('g++', ['-O2', '-std=c++17', '-I', path.join(repo, 'gpu'), path.join(here, 'vkhdr.cpp'),
    path.join(repo, 'gpu', 'build-webgl', 'libframegen.a'), '-lvulkan', '-o', vkhdr], { stdio: 'inherit' });
  if (r.status !== 0) { console.error('hdr: building vkhdr failed (run vkref.mjs --build first for libframegen.a)'); process.exit(2); }
}
const man = JSON.parse(fs.readFileSync(path.join(hdir, 'manifest.json'), 'utf8'));
const SET = { v1: ['algo=v1', { motionVersion: 1 }], v2p: ['algo=v2,v2Mode=performance', { motionVersion: 2, v2Mode: 'performance' }],
  v2q: ['algo=v2,v2Mode=quality', { motionVersion: 2, v2Mode: 'quality' }] };

// Vulkan references (RgbaF16)
const refNote = {};
for (const p of man.pairs) {
  if (opt.pairs && !opt.pairs.includes(p.name)) continue;
  if (p.enc === 'id') continue;   // compared against the Rgba8 reference of vkref.mjs
  for (const mode of opt.modes) for (const S of opt.fs) for (const t of man.ts) {
    const d = path.join(hdir, 'vk', `${mode}_fs${S}`, p.name);
    const base = path.join(d, 't' + t);
    if (fs.existsSync(base + '.stats')) continue;
    fs.mkdirSync(d, { recursive: true });
    const r = spawnSync(vkhdr, ['--set', `${SET[mode][0]},flowScale=${S}`, '--t', String(t), '--format', 'f16', '--transfer', p.transfer, '--white', '80',
      String(p.width), String(p.height), path.join(hdir, p.name, 'a.f16'), path.join(hdir, p.name, 'c.f16'), base + '.f16'], { encoding: 'utf8' });
    const st = (r.stdout || '').split('\n').find((l) => l.startsWith('STATS'));
    if (r.status !== 0 || !st) { refNote[`${mode}_fs${S}`] = (r.stderr || '').trim().split('\n').pop(); continue; }
    fs.writeFileSync(base + '.stats', st + '\n');
  }
}
for (const [k, v] of Object.entries(refNote)) console.log(`vkhdr ${k}: no Vulkan RgbaF16 reference: ${v}`);

const E = await openEngine({ settings: { debug: true } });
console.log(`browser : Chromium ${E.version}, renderer ${E.init.info && E.init.info.renderer}`);
if (E.init.fatal) { console.error('hdr: engine init failed: ' + E.init.fatal); await E.browser.close(); process.exit(2); }
const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : String(v));
const pad = (s, n) => String(s).padEnd(n);
let allOk = true;
const rows = [];
for (const mode of opt.modes) for (const S of opt.fs) for (const p of man.pairs) {
  if (opt.pairs && !opt.pairs.includes(p.name)) continue;
  const cfg = `${mode}_fs${S}`;
  const refs = {};
  for (const t of man.ts) {
    if (p.enc === 'id') refs[t] = { file: `vkref/${cfg}/${p.src}/t${t}.rgba`, fmt: 'rgba8', stats: `vkref/${cfg}/${p.src}/t${t}.stats` };
    else if (fs.existsSync(path.join(hdir, 'vk', cfg, p.name, `t${t}.stats`))) refs[t] = { file: `hdr/vk/${cfg}/${p.name}/t${t}.f16`, fmt: 'f16', stats: `hdr/vk/${cfg}/${p.name}/t${t}.stats` };
  }
  const settings = { ...SET[mode][1], flowScale: S, inputFormat: 'rgba16f', transfer: p.transfer, linearWhiteNits: 80 };
  const r = await E.page.evaluate((j) => window.__x.runHdr(j), { dir: `hdr/${p.name}`, W: p.width, H: p.height, settings, ts: man.ts, refs });
  if (r.error) { console.log(`${cfg} ${p.name}: ENGINE ERROR ${r.error.error}\n${r.error.stack}`); allOk = false; continue; }
  for (const t of man.ts) {
    const T = r.t[t];
    let ok = T.synth && T.nonFinite === 0 && !r.glError, dec = '', cmp = 'no Vulkan reference (finite output only)';
    if (T.cmp) {
      const [libUntr, libSad256, libBlocks] = T.libStats;
      const libCut = libSad256 / 256 / libBlocks > 30, libFlick = libUntr > 0.15 * p.width * p.height;
      const decOk = T.stats.cut === libCut && (libCut || T.stats.flick === libFlick);
      const fb = libCut || libFlick;
      const imgOk = fb ? T.cmp.max < 0.5 : T.cmp.mae <= TOL.mae && T.cmp.pct4 <= TOL.pct4;
      ok = ok && decOk && imgOk;
      dec = `dec ${decOk ? 'same' : 'DIFF'}${fb ? ' (' + (libCut ? 'cut' : 'flick') + ')' : ''} sad ${f3(T.stats.meanSad)}/${f3(libSad256 / 256 / libBlocks)} untr ${T.stats.untrusted}/${libUntr}`;
      cmp = `MAE ${f3(T.cmp.mae)} max ${f3(T.cmp.max)} %>4 ${f3(T.cmp.pct4)} vs ${p.enc === 'id' ? 'Vulkan Rgba8' : 'Vulkan RgbaF16'}`;
    }
    if (!ok) allOk = false;
    rows.push({ cfg, pair: p.name, t, ok, cmp: T.cmp || null, hasRef: !!T.cmp });
    console.log(`${pad(cfg, 8)} ${pad(p.name, 16)} t=${pad(t, 5)} ${ok ? 'PASS' : 'FAIL'}  ${cmp}  ${dec}  %>1.0 ${f3(T.pctOver1)}${T.nonFinite ? ' NON-FINITE ' + T.nonFinite : ''}`);
  }
}
if (E.consoleMsgs.size) { console.log('\npage console:'); for (const [k, v] of E.consoleMsgs) console.log(`  [x${v}] ${k}`); }
await E.browser.close();
console.log(`RESULT: ${allOk ? 'PASS' : 'FAIL'} (${rows.filter((r) => r.hasRef).length} of ${rows.length} rows against a Vulkan reference)`);
process.exit(allOk ? 0 : 1);
