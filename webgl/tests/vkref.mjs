#!/usr/bin/env node
// Vulkan reference of the cross-backend check: runs the library (gpu/, its test tool fgtest, on a real Vulkan GPU) on every pair of
// $FG_WORK/pairs (pairs.py) for each mode x flowScale x t, once per t (so --stats holds that synth's decision inputs).
//   node webgl/tests/vkref.mjs [--build] [--modes v1,v2p,v2q] [--pairs a,b] [--static 0|1] [--force]
// Env FGTEST: a built fgtest (default <repo>/gpu/build-webgl/fgtest; --build = meson setup + ninja there, gitignored as gpu/build*/).
// Output: $FG_WORK/vkref/<mode>[_s0]_fs<S>/<pair>/t<T>.rgba (W*H*4), .stats ("STATS <pair> <untrusted px> <SAD sum x256> <blocks>
// <W*H>"), .fgmaps (fgtest --dump: the block field, V2 Quality's per-pixel field and coverage; gpu/fgmaps.py layout).
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { repo, work } from './paths.mjs';

const argv = process.argv.slice(2);
const opt = { build: false, modes: ['v1', 'v2p', 'v2q'], pairs: null, staticMask: 1, force: false };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], nx = () => argv[++i];
  if (a === '--build') opt.build = true;
  else if (a === '--modes') opt.modes = nx().split(',');
  else if (a === '--pairs') opt.pairs = nx().split(',');
  else if (a === '--static') opt.staticMask = Number(nx());
  else if (a === '--force') opt.force = true;
  else { console.error('vkref: unknown argument ' + a); process.exit(2); }
}
const bdir = path.join(repo, 'gpu', 'build-webgl');
if (opt.build) {
  if (!fs.existsSync(path.join(bdir, 'build.ninja'))) spawnSync('meson', ['setup', bdir, path.join(repo, 'gpu')], { stdio: 'inherit' });
  const r = spawnSync('ninja', ['-C', bdir, 'fgtest'], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(2);
}
const fgtest = process.env.FGTEST || path.join(bdir, 'fgtest');
if (!fs.existsSync(fgtest)) { console.error(`vkref: fgtest not found at ${fgtest} (set FGTEST or use --build)`); process.exit(2); }
const pdir = path.join(work, 'pairs');
const man = JSON.parse(fs.readFileSync(path.join(pdir, 'manifest.json'), 'utf8'));
export const MODE_SET = {
  v1: 'algo=v1',
  v2p: 'algo=v2,v2Mode=performance',
  v2q: 'algo=v2,v2Mode=quality',
};
let n = 0;
for (const p of man.pairs) {
  if (opt.pairs && !opt.pairs.includes(p.name)) continue;
  for (const mode of opt.modes) {
    for (const S of p.flow_scales) {
      const cfg = `${mode}${mode !== 'v1' && !opt.staticMask ? '_s0' : ''}_fs${S}`;
      const d = path.join(work, 'vkref', cfg, p.name);
      fs.mkdirSync(d, { recursive: true });
      for (const t of man.ts) {
        const base = path.join(d, 't' + t);
        if (!opt.force && fs.existsSync(base + '.stats') && fs.existsSync(base + '.rgba')) continue;
        const set = `${MODE_SET[mode]}${mode === 'v1' ? '' : ',v2StaticMask=' + opt.staticMask},flowScale=${S}`;
        const dump = base + '_dump';
        const r = spawnSync(fgtest, ['--set', set, '--t', String(t), '--stats', '--dump', dump, String(p.width), String(p.height),
          path.join(pdir, p.name, 'a.rgba'), path.join(pdir, p.name, 'c.rgba'), base + '.rgba'], { encoding: 'utf8' });
        const st = (r.stdout || '').split('\n').find((l) => l.startsWith('STATS'));
        if (r.status !== 0 || !st) { console.error(`vkref: fgtest failed for ${cfg} ${p.name} t=${t}: ${r.stderr}`); process.exit(1); }
        fs.writeFileSync(base + '.stats', st + '\n');
        fs.renameSync(path.join(dump, 'pair_0000.fgmaps'), base + '.fgmaps');
        fs.rmdirSync(dump);
        n++;
        console.log(`vkref: ${cfg} ${p.name} ${p.width}x${p.height} t=${t}: ${st}`);
      }
    }
  }
}
console.log(`vkref: ${n} runs written under ${path.join(work, 'vkref')}`);
