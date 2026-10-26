#!/usr/bin/env node
// Cost per real pair (push the new frame + estimate + synth(0.5) = 2x) of each motion version in headless Chromium.
//   node webgl/tests/bench.mjs [--gpu] [--pair battlefield] [--modes v1,v2p,v2q,ml2,ml1,ml2f,ml1f] [--fs 2] [--iters 60] [--mult 2] [--size WxH]
// --passes: also GPU time per pass (timer query around every draw).
// --gpu: hardware GL through ANGLE (FG_ANGLE=vulkan default, or gl); otherwise SwiftShader (CPU: only relative numbers mean much).
// Modes: v1, v2p (V2 Performance), v2q (V2 Quality), ml2 / ml1 (V1 with the learned estimator e2v3 / e1v3, motionPrecision fp16;
// ml2f / ml1f: fp32; flowScale 2 only), each with the zero-motion test (the default); a suffix n turns the zero test off (ml2n,
// ml2fn, ...), a suffix g turns the static guard on (ml2g, ml2ng, ...). Any net by its name the same way: e2v4 (fp16), e2v4f (fp32),
// e2v4n, e2v4fg, ... (ml2 = e2v3, ml1 = e1v3; webgl/nets/<net>.js must exist).
// --size: the pair resampled (bilinear) to W x H first, e.g. 2560x1440 from the 1080p pair.
// GPU time (EXT_disjoint_timer_query_webgl2) of pushFrame (the raw RGBA upload from JS + luma pyramid) and of estimate + synths, when the browser exposes it, else wall time per pair with a gl.finish() each
// (includes the CPU side of the GL calls). Prints the median and the minimum over --iters pairs after 10 warm-up pairs.
import fs from 'node:fs';
import path from 'node:path';
import { openEngine } from './server.mjs';
import { work } from './paths.mjs';

const argv = process.argv.slice(2);
const opt = { passes: false, gpu: false, pair: 'battlefield', modes: ['v1', 'v2p', 'v2q'], fs: [2], iters: 60, mult: 2, size: null };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], nx = () => argv[++i];
  if (a === '--gpu') opt.gpu = true;
  else if (a === '--passes') opt.passes = true;
  else if (a === '--pair') opt.pair = nx();
  else if (a === '--modes') opt.modes = nx().split(',');
  else if (a === '--fs') opt.fs = nx().split(',').map(Number);
  else if (a === '--iters') opt.iters = Number(nx());
  else if (a === '--mult') opt.mult = Number(nx());
  else if (a === '--size') opt.size = nx().split('x').map(Number);
  else { console.error('bench: unknown argument ' + a); process.exit(2); }
}
const man = JSON.parse(fs.readFileSync(path.join(work, 'pairs', 'manifest.json'), 'utf8'));
const p = man.pairs.find((x) => x.name === opt.pair);
if (!p) { console.error('bench: no pair ' + opt.pair); process.exit(2); }
const E = await openEngine({ gpu: opt.gpu, settings: {} });
if (E.init.fatal) { console.error('bench: ' + E.init.fatal); await E.browser.close(); process.exit(2); }
console.log(`browser : Chromium ${E.version}, renderer ${E.init.info.renderer}`);
const SET = { v1: { motionVersion: 1, motion: 'search' }, v2p: { motionVersion: 2, v2Mode: 'performance' }, v2q: { motionVersion: 2, v2Mode: 'quality' },
  ml2: { motionVersion: 1, motion: 'learned', motionNet: 'e2v3', motionPrecision: 'fp16' }, ml1: { motionVersion: 1, motion: 'learned', motionNet: 'e1v3', motionPrecision: 'fp16' },
  ml2f: { motionVersion: 1, motion: 'learned', motionNet: 'e2v3', motionPrecision: 'fp32' }, ml1f: { motionVersion: 1, motion: 'learned', motionNet: 'e1v3', motionPrecision: 'fp32' } };
const setOf = (mode) => {   // (every learned mode sets both switches: settings persist across modes)
  const m = /^(ml[12]f?)(n?)(g?)$/.exec(mode), k = /^(e\d+v\d+)(f?)(n?)(g?)$/.exec(mode);
  if (k) return { motionVersion: 1, motion: 'learned', motionNet: k[1], motionPrecision: k[2] ? 'fp32' : 'fp16', learnedZeroTest: !k[3], learnedStaticGuard: !!k[4] };
  if (!m) return SET[mode];
  return { ...SET[m[1]], learnedZeroTest: !m[2], learnedStaticGuard: !!m[3] };
};
await E.page.addScriptTag({ content: `
window.__bench = async function (job) {
  const gl = window.__x._gl();
  const eng = window.__x._engine();
  const get = async (q) => new Uint8Array(await (await fetch('/w/' + q)).arrayBuffer());
  let a = await get('pairs/' + job.pair + '/a.rgba'), c = await get('pairs/' + job.pair + '/c.rgba');
  if (job.size) {   // bilinear resample (half-pixel centres) to job.size
    const rs = (src) => {
      const [w2, h2] = job.size, o = new Uint8Array(w2 * h2 * 4), sx = job.W0 / w2, sy = job.H0 / h2;
      for (let y = 0; y < h2; y++) {
        const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), job.H0 - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, job.H0 - 1), wy = fy - y0;
        for (let x = 0; x < w2; x++) {
          const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), job.W0 - 1), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, job.W0 - 1), wx = fx - x0;
          for (let k = 0; k < 4; k++) {
            const v = (src[(y0 * job.W0 + x0) * 4 + k] * (1 - wx) + src[(y0 * job.W0 + x1) * 4 + k] * wx) * (1 - wy) +
              (src[(y1 * job.W0 + x0) * 4 + k] * (1 - wx) + src[(y1 * job.W0 + x1) * 4 + k] * wx) * wy;
            o[(y * w2 + x) * 4 + k] = Math.round(v);
          }
        }
      }
      return o;
    };
    a = rs(a); c = rs(c);
  }
  eng.setSettings(job.settings);
  const tq = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  const fr = [{ width: job.W, height: job.H, data: a }, { width: job.W, height: job.H, data: c }];
  const ts = []; for (let k = 1; k < job.mult; k++) ts.push(k / job.mult);
  const one = () => { eng.pushFrame(fr[0]); eng.pushFrame(fr[1]); eng.estimate(); for (const t of ts) eng.synth(t, undefined); };
  const px = new Uint8Array(4);
  const sync = () => { gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); };
  for (let i = 0; i < 10; i++) one();
  sync();
  const wall = [], gpu = [], gpuPush = [];
  const timed = async (f, out) => {
    let q = null;
    if (tq) { q = gl.createQuery(); gl.beginQuery(tq.TIME_ELAPSED_EXT, q); }
    f();
    if (tq) gl.endQuery(tq.TIME_ELAPSED_EXT);
    return q;
  };
  const result = async (q, out) => {
    if (!q) return;
    let n = 0;
    while (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE) && n++ < 1000) await new Promise((r) => setTimeout(r, 1));
    if (!gl.getParameter(tq.GPU_DISJOINT_EXT)) out.push(gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
    gl.deleteQuery(q);
  };
  for (let i = 0; i < job.iters; i++) {
    eng.pushFrame(fr[0]);   // the older frame (not timed): a pair costs one push (the new frame) + estimate + synths
    sync();
    const t0 = performance.now();
    const q1 = await timed(() => eng.pushFrame(fr[1]));   // upload + luma pyramid
    const q2 = await timed(() => { eng.estimate(); for (const t of ts) eng.synth(t, undefined); });
    sync();
    wall.push(performance.now() - t0);
    await result(q1, gpuPush);
    await result(q2, gpu);
  }
  // back-to-back: 20 estimate + synths of the same pair in one query (GPU kept busy, no per-pair sync)
  const batch = [];
  for (let r = 0; r < 5; r++) {
    sync();
    const q = await timed(() => { for (let k = 0; k < 20; k++) { eng.estimate(); for (const t of ts) eng.synth(t, undefined); } });
    sync();
    const o = []; await result(q, o); if (o.length) batch.push(o[0] / 20);
  }
  const batchE = [], batchS = [];
  for (let r = 0; r < 5; r++) {
    sync();
    const qe = await timed(() => { for (let k = 0; k < 20; k++) eng.estimate(); });
    sync();
    const o = []; await result(qe, o); if (o.length) batchE.push(o[0] / 20);
    const qs = await timed(() => { for (let k = 0; k < 20; k++) for (const t of ts) eng.synth(t, undefined); });
    sync();
    const o2 = []; await result(qs, o2); if (o2.length) batchS.push(o2[0] / 20);
  }
  // --passes: GPU time per program (one timer query around each draw), summed per pair, median over 10 pairs
  let perPass = null;
  if (job.passes && tq) {
    const acc = {};
    for (let r = 0; r < 10; r++) {
      const qs = [];
      let cur = null;
      eng.setPassHook((name, ph) => {
        if (ph === 0) { cur = gl.createQuery(); gl.beginQuery(tq.TIME_ELAPSED_EXT, cur); }
        else { gl.endQuery(tq.TIME_ELAPSED_EXT); qs.push([name, cur]); }
      });
      eng.pushFrame(fr[0]); eng.pushFrame(fr[1]); eng.estimate(); for (const t of ts) eng.synth(t, undefined);
      eng.setPassHook(null);
      sync();
      const sums = {};
      for (const [name, q] of qs) { const o = []; await result(q, o); sums[name] = (sums[name] || 0) + (o[0] || 0); }
      for (const [k, v] of Object.entries(sums)) (acc[k] = acc[k] || []).push(v);
    }
    perPass = Object.fromEntries(Object.entries(acc).map(([k, v]) => [k, v.sort((x, y) => x - y)[v.length >> 1]]));
  }
  const med = (v) => { const s = v.slice().sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
  return { wall: med(wall), wallMin: Math.min(...wall), push: med(gpuPush), gpu: med(gpu), batch: med(batch), bE: med(batchE), bS: med(batchS), perPass, gpuMin: gpu.length ? Math.min(...gpu) : null, timer: !!tq, passCounts: eng.passCounts() };
};` });
for (const mode of opt.modes) if (!setOf(mode)) { console.error('bench: unknown mode ' + mode); process.exit(2); }
for (const mode of opt.modes) for (const S of opt.fs) {
  const W = opt.size ? opt.size[0] : p.width, H = opt.size ? opt.size[1] : p.height;
  const r = await E.page.evaluate((j) => window.__bench(j), { pair: p.name, W, H, W0: p.width, H0: p.height, size: opt.size, settings: { ...setOf(mode), flowScale: S }, passes: opt.passes, iters: opt.iters, mult: opt.mult });
  const f = (v) => (v == null ? '-' : v.toFixed(3));
  if (r.perPass) console.log('  per pass (ms): ' + Object.entries(r.perPass).sort((x, y) => y[1] - x[1]).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(', '));
  console.log(`${mode.padEnd(5)} fs${S} ${p.name} ${W}x${H} ${opt.mult}x: wall ${f(r.wall)} ms (min ${f(r.wallMin)})  gpu: push ${f(r.push)} ms + estimate/synth ${f(r.gpu)} ms (min ${f(r.gpuMin)}), back-to-back ${f(r.batch)} ms (estimate ${f(r.bE)}, synth ${f(r.bS)})${r.timer ? '' : ' [no timer query]'}  passes ${JSON.stringify(r.passCounts)}`);
}
await E.browser.close();
