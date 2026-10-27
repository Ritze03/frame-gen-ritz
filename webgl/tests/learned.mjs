#!/usr/bin/env node
// The learned motion estimator (settings.motion 'learned', webgl/shaders/learned/) vs the ml-fg hand-off test vectors, by the
// procedure of export/SPEC.md section 8, for every pair of every net at both activation precisions; then the duplicate-pair skip.
//   node webgl/tests/learned.mjs [--gpu] [--nets e2v5,e1v5,e2v4,e1v4,e2v3,e1v3] [--pairs motion,odd] [--prec fp32,fp16] [--json FILE]
// Nets: each webgl/nets/<net>.js with an export/<net>/testvec; pairs: its testvec/<pair>/ directories (the cut flags are vec.json's
// cut / cut_zt, never the pair's name).
// Vectors: $FG_MLFG_EXPORT (default ~/Schinken/Programming/Local/ml-fg/export; read only, nothing is copied into the repo); the decoded
// frames, the real middle frames ($FG_FOOTAGE) and the engine's dumps go to $FG_WORK/learned (learned_ref.py prep / synth).
// --gpu: hardware GL, else SwiftShader.
// Gates per net x pair x precision:
//   both: ya / yc exact (the net's own cv2 5.0.0 luma); the same cut decision; synth(engine pf) vs synth(expected pf) through
//     proto/interp.py >= 45 dB at t = 0.5 and 0.25 (an exact port reaches >= 80 dB); where the real middle frame is known, the PSNR
//     of synth(engine pf) against it within 0.05 dB of synth(expected pf)'s; the engine's own output (V1 synth on its field) vs
//     synth(expected pf) MAE <= 1 level (golden's gate); not taken for a duplicate.
//   motionPrecision 'fp32' (the exact float32 port, SPEC section 8's tolerances = reference.py's constants): coarsest level in_top,
//     act_top0..3, raw_top max |d| <= 5e-5; pf mean |d| <= 2e-4 px and max |d| <= 5e-2 px; sad relative error <= 1e-3.
//   motionPrecision 'fp16' (half-float activations, SPEC section 7: judged by the synthesised frame): sad relative error <= 1e-2;
//     pf and the intermediates are reported, not gated.
//   The rows above run with settings.learnedZeroTest false (the net's own field).
// Zero-motion test (settings.learnedZeroTest true, SPEC stage 7, the vectors' *_zt files), the same pair estimated again: the net's
// field (ml.pfNet) bit-identical to the run without the test; the same cut_zt; synth(final pf) vs synth(pf_zt) at t = 0.5 / 0.25
// >= 70 dB for 'fp32' (SPEC section 8's gate for an exact port), >= 45 dB otherwise; vs the real middle frame within 0.05 dB of
// synth(pf_zt)'s; engine output vs synth(pf_zt) MAE <= 1 level.
//   'fp32': zero-mask agreement with zero_zt >= 99.9%; pf vs pf_zt on the pixels where the masks agree mean <= 2e-4 px, max <= 5e-2
//     px; sad_zt relative error <= 1e-3; C(0) (independent of the field) max |d| <= 2e-3 levels against cost0_zt.
//   'fp16' / 'hdr': sad_zt relative error <= 1e-2; the mask agreement, pf and costs are reported, not gated.
// Duplicate skip (per net x precision x zero test on / off): a == c gives pixfield 0, sad 0, duplicate true, and an output identical
// to the input at t = 0.5 and 0.25; a near-duplicate (+-1 level noise, mean |a - c| < 0.6) is skipped too; the test pairs are not.
// Static guard (settings.learnedStaticGuard; e2v3 fp32 with the zero test, pairs overlay / motion and the first with cut_zt true):
// the guarded output vs ml-fg's scripts/static_guard.py on the engine's own a, c, unguarded output and field at t: max |d| <= 1 level
// (RGBA8 and RGBA16F input); the guard switched off again gives the unguarded output bit for bit; a fallback frame (cut) is unchanged.
// Exit 0 PASS, 1 FAIL, 2 setup error.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { openEngine } from './server.mjs';
import { here, work } from './paths.mjs';

const argv = process.argv.slice(2);
const opt = { gpu: false, nets: null, pairs: null, json: null, prec: ['fp32', 'fp16'] };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], nx = () => argv[++i];
  if (a === '--gpu') opt.gpu = true;
  else if (a === '--nets') opt.nets = nx().split(',');
  else if (a === '--pairs') opt.pairs = nx().split(',');
  else if (a === '--prec') opt.prec = nx().split(',');
  else if (a === '--json') opt.json = nx();
  else { console.error('learned: unknown argument ' + a); process.exit(2); }
}
const TOL = { top: 5e-5, pfMean: 2e-4, pfMax: 5e-2, sadRel: 1e-3, sadRel16: 1e-2, synthDb: 45, midDb: 0.05, engineMae: 1.0,
  ztAgree: 0.999, ztSynthDb: 70, cost: 2e-3, sgMax: 1.0 };
const HDR_NETS = ['e2v5', 'e2v4', 'e2v3'];   // the RGBA16F-input rows (motion / odd, fp16)
const SG_NET = 'e2v3';               // the static guard check's net (pinned: the guard does not depend on the net)
const py = (mode) => {
  const r = spawnSync('python3', ['-I', path.join(here, 'learned_ref.py'), mode], { stdio: ['ignore', 'pipe', 'inherit'], encoding: 'utf8', env: { ...process.env, FG_WORK: work } });
  if (r.status !== 0) { console.error(`learned: learned_ref.py ${mode} failed (${r.status})`); process.exit(2); }
  return r.stdout;
};
py('prep');
const man = JSON.parse(fs.readFileSync(path.join(work, 'learned', 'manifest.json'), 'utf8'));
console.log(`vectors : ${man.export}`);
const E = await openEngine({ gpu: opt.gpu, settings: { debug: true } });
console.log(`browser : Chromium ${E.version}, renderer ${E.init.info && E.init.info.renderer}`);
if (E.init.fatal) { console.error('learned: engine init failed: ' + E.init.fatal); await E.browser.close(); process.exit(2); }
console.log(`support : ${JSON.stringify(E.init.info.support)}`);

await E.page.addScriptTag({ content: `
(() => {
  const err = (e) => ({ error: String((e && e.message) || e), stack: e && e.stack ? String(e.stack).split('\\n').slice(0, 6).join('\\n') : '' });
  const get = async (q) => new Uint8Array(await (await fetch('/w/' + q)).arrayBuffer());
  const post = (name, arr) => fetch('/dump/' + name, { method: 'POST', body: arr });
  const half = (b) => {   // IEEE half bits -> number
    const s = (b & 0x8000) ? -1 : 1, e = (b >> 10) & 31, f = b & 1023;
    return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  };
  // the channels of the textures \`names\` (RGBA / RG float: 4 channels per texture; RGBA32UI: 8 packed halves, motionPrecision
  // 'fp16'), cropped to w x h, channel-first [nch][h][w]
  function chw(eng, names, nch, w, h) {
    const out = new Float32Array(nch * w * h);
    let base = 0;
    for (const n of names) {
      const t = eng.debugRead(n), packed = t.data instanceof Uint32Array, per = packed ? 8 : 4;
      for (let c = 0; c < per && base + c < nch; c++)
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          const i = (y * t.width + x) * t.channels;
          out[((base + c) * h + y) * w + x] = packed ? half((t.data[i + (c >> 1)] >>> (16 * (c & 1))) & 0xffff) : c < t.channels ? t.data[i + c] : 0;
        }
      base += per;
    }
    return out;
  }
  window.__learned = async function (job) {
    const R = { t: {} };
    const eng = window.__x._engine();
    try {
      let a = await get(job.dir + '/a.rgba'), c = await get(job.dir + '/c.rgba');
      R.setSettings = eng.setSettings({ motionVersion: 1, motion: 'learned', motionNet: job.net, motionPrecision: job.prec, flowScale: 2,
        learnedZeroTest: false, learnedStaticGuard: false, inputFormat: job.f16in ? 'rgba16f' : 'rgba8', transfer: 'encoded' });
      if (job.f16in) { a = Float32Array.from(a, (v) => v / 255); c = Float32Array.from(c, (v) => v / 255); }
      eng.pushFrame({ width: job.W, height: job.H, data: a });
      eng.pushFrame({ width: job.W, height: job.H, data: c });
      const net = __fg.nets[job.net], G = net.width / 4, top = net.levels - 1;
      const tw = job.padded[0] >> top, th = job.padded[1] >> top;
      const fp16 = job.prec === 'fp16', sfx = G === 6 && !fp16 ? 'b' : '';   // (fp32, width 24: a layer is passes a and b)
      const nx = fp16 ? 2 : 4, nAct = fp16 ? Math.ceil(G / 2) : G;
      const cap = {};
      eng.setPassHook((name, ph) => {
        if (ph !== 1) return;
        if (name === 'ml_match.' + top) cap.in_top = chw(eng, Array.from({ length: nx }, (_, k) => 'ml.x.' + k), 14, tw, th);
        for (let k = 0; k < net.depth; k++) {
          if (name === 'ml_c' + k + sfx + '.' + top) cap['act_top' + k] = chw(eng, Array.from({ length: nAct }, (_, g) => 'ml.act.' + (k & 1) + '.' + g), net.width, tw, th);
        }
      });
      R.estimate = eng.estimate();
      eng.setPassHook(null);
      R.passCounts = eng.passCounts();
      const st = eng.stats();
      R.sad = st.meanSad; R.cut = st.cut; R.duplicate = st.duplicate; R.dupDiff = st.dupDiff;
      const pf = eng.debugRead('pixfield');
      await post(job.tag + '/pf', pf.data);
      for (let l = 1; l < net.levels; l++) await post(job.tag + '/h_level' + l, eng.debugRead('ml.h.' + l).data);
      const h4 = eng.debugRead('ml.h.' + top), raw = new Float32Array(2 * tw * th);
      for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) for (let k = 0; k < 2; k++) raw[(k * th + y) * tw + x] = h4.data[(y * tw + x) * 2 + k] / 2;
      cap.raw_top = raw;   // (h = 0 before the top level: raw = (h + 2 raw) / 2)
      for (const [k, v] of Object.entries(cap)) await post(job.tag + '/' + k, v);
      const lv0 = eng.debugRead('ml.pyr.0'), fw = job.flow[0], fh = job.flow[1];
      const ya = new Float32Array(fw * fh), yc = new Float32Array(fw * fh);
      for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) { const i = (y * lv0.width + x) * 4; ya[y * fw + x] = lv0.data[i + 2]; yc[y * fw + x] = lv0.data[i + 3]; }
      await post(job.tag + '/ya', ya); await post(job.tag + '/yc', yc);
      for (const t of [0.5, 0.25]) {
        R.t[t] = eng.synth(t, undefined);
        if (R.t[t]) {
          const o = eng.debugRead('out').data;   // (rgba16f: values -> levels)
          await post(job.tag + '/out_t' + t, job.f16in ? Uint8Array.from(o, (v) => Math.min(255, Math.max(0, Math.round(v * 255)))) : o);
        }
        R['stats' + t] = eng.stats();
      }
      // the zero-motion test on the same pair (the setting change drops the estimate; the frames stay)
      eng.setSettings({ learnedZeroTest: true });
      const Z = R.zt = { t: {} };
      Z.estimate = eng.estimate();
      const pfNet = eng.debugRead('ml.pfNet').data;
      Z.pfNetDiff = 0;
      for (let i = 0; i < pfNet.length; i++) if (pfNet[i] !== pf.data[i]) Z.pfNetDiff++;
      const sz = eng.stats();
      Z.sad = sz.meanSad; Z.cut = sz.cut; Z.duplicate = sz.duplicate;
      await post(job.tag + '/zt_pf', eng.debugRead('pixfield').data);
      await post(job.tag + '/zt_zt', eng.debugRead('ml.zt').data);
      for (const t of [0.5, 0.25]) {
        Z.t[t] = eng.synth(t, undefined);
        if (Z.t[t]) {
          const o = eng.debugRead('out').data;
          await post(job.tag + '/zt_out_t' + t, job.f16in ? Uint8Array.from(o, (v) => Math.min(255, Math.max(0, Math.round(v * 255)))) : o);
        }
      }
      Z.passCounts = eng.passCounts();
      eng.setSettings({ learnedZeroTest: false });
      if (job.f16in) eng.setSettings({ inputFormat: 'rgba8' });
      R.glError = window.__x._gl().getError();
    } catch (e) { R.error = err(e); }
    return R;
  };
  // the duplicate skip: a == c (and a near-duplicate) in learned mode
  window.__dupTest = async function (job) {
    const R = {};
    const eng = window.__x._engine();
    try {
      const a = await get(job.dir + '/a.rgba');
      eng.setSettings({ motionVersion: 1, motion: 'learned', motionNet: job.net, motionPrecision: job.prec, flowScale: 2, learnedZeroTest: job.zt });
      const zeroField = () => { const f = eng.debugRead('pixfield').data; let n = 0; for (let i = 0; i < f.length; i++) if (f[i] !== 0) n++; return n; };
      const one = (c) => {
        const r = {};
        eng.pushFrame({ width: job.W, height: job.H, data: a });
        eng.pushFrame({ width: job.W, height: job.H, data: c });
        r.estimate = eng.estimate();
        r.nonZeroField = zeroField();
        for (const t of [0.5, 0.25]) {
          r['synth' + t] = eng.synth(t, undefined);
          const o = eng.debugRead('out').data;
          let diff = 0, mx = 0;
          for (let i = 0; i < o.length; i++) { if ((i & 3) === 3) continue; const d = Math.abs(o[i] - a[i]); if (d) diff++; if (d > mx) mx = d; }
          r['diffVsA' + t] = diff; r['maxVsA' + t] = mx;
        }
        const s = eng.stats();
        r.duplicate = s.duplicate; r.dupDiff = s.dupDiff; r.meanSad = s.meanSad; r.cut = s.cut; r.fallback = s.fallback;
        return r;
      };
      R.identical = one(a);
      const nd = a.slice();
      let k = 12345;
      for (let i = 0; i < nd.length; i++) { if ((i & 3) === 3) continue; k = (k * 1103515245 + 12345) >>> 0; const r = k >>> 29; if (r === 0 && nd[i] < 255) nd[i]++; else if (r === 1 && nd[i] > 0) nd[i]--; }
      R.near = one(nd);
      R.glError = window.__x._gl().getError();
    } catch (e) { R.error = err(e); }
    return R;
  };
  // the static guard on one pair: per t the unguarded output, the field at t and the guarded output (dumped for learned_ref.py
  // sguard), then the guard off again (must give the unguarded output)
  window.__sgTest = async function (job) {
    const R = { t: {} };
    const eng = window.__x._engine();
    try {
      let a = await get(job.dir + '/a.rgba'), c = await get(job.dir + '/c.rgba');
      eng.setSettings({ motionVersion: 1, motion: 'learned', motionNet: job.net, motionPrecision: job.prec, flowScale: 2, learnedZeroTest: true,
        learnedStaticGuard: false, inputFormat: job.f16in ? 'rgba16f' : 'rgba8', transfer: 'encoded' });
      if (job.f16in) { a = Float32Array.from(a, (v) => v / 255); c = Float32Array.from(c, (v) => v / 255); }
      eng.pushFrame({ width: job.W, height: job.H, data: a });
      eng.pushFrame({ width: job.W, height: job.H, data: c });
      R.estimate = eng.estimate();
      for (const t of job.ts) {
        const r = R.t[t] = {};
        const out = () => { r.ok = eng.synth(t, undefined) && (r.ok !== false); return eng.debugRead('out').data.slice(); };
        eng.setSettings({ learnedStaticGuard: false });
        const o0 = out();
        const field = eng.debugRead(t === 0.5 ? 'pixfield' : 'fieldT').data;
        eng.setSettings({ learnedStaticGuard: true });
        const og = out();
        r.passCounts = eng.passCounts();
        r.fallback = eng.stats().fallback;
        eng.setSettings({ learnedStaticGuard: false });
        const o1 = out();
        r.offDiff = 0;
        for (let i = 0; i < o0.length; i++) if (o0[i] !== o1[i]) r.offDiff++;
        r.guardDiff = 0;
        for (let i = 0; i < o0.length; i++) if (o0[i] !== og[i]) r.guardDiff++;
        await post(job.tag + '/' + t + '/out0', o0); await post(job.tag + '/' + t + '/outg', og); await post(job.tag + '/' + t + '/field', field);
      }
      if (job.f16in) eng.setSettings({ inputFormat: 'rgba8' });
      R.glError = window.__x._gl().getError();
    } catch (e) { R.error = err(e); }
    return R;
  };
})();` });

const rdF32 = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const cmp = (e, x) => {
  let mx = 0, s = 0, neq = 0;
  for (let i = 0; i < x.length; i++) { const d = Math.abs(e[i] - x[i]); if (!(d <= mx)) mx = Number.isNaN(d) ? Infinity : d; s += d; if (d !== 0) neq++; }
  return { max: mx, mean: s / x.length, neq };
};
const e1 = (v) => (v === 0 ? '0' : v.toExponential(1));
const rows = [];
let allOk = true;
// jobs: every pair at each precision, plus (variant 'hdr') e2v5 / e2v4 / e2v3 motion / odd with RGBA16F input (inputFormat 'rgba16f', transfer
// 'encoded': the frames as value / 255; the luma is then the unrounded weighted sum, so ya / yc are not compared)
const jobs = [];
for (const prec of opt.prec) for (const p of man.pairs) jobs.push({ p, prec, variant: prec });
if (opt.prec.includes('fp16')) for (const p of man.pairs) if (HDR_NETS.includes(p.net) && ['motion', 'odd'].includes(p.pair)) jobs.push({ p, prec: 'fp16', variant: 'hdr' });
for (const { p, prec, variant } of jobs) {
  {
    if (opt.nets && !opt.nets.includes(p.net)) continue;
    if (opt.pairs && !opt.pairs.includes(p.pair)) continue;
    const tag = `${p.net}_${p.pair}_${variant}`, f16in = variant === 'hdr';
    const r = await E.page.evaluate((j) => window.__learned(j), { net: p.net, prec, f16in, dir: `learned/${p.net}/${p.pair}`, W: p.W, H: p.H, flow: p.flow, padded: p.padded, tag });
    if (r.error) { console.log(`${tag}: ENGINE ERROR ${r.error.error}\n${r.error.stack}`); allOk = false; rows.push({ net: p.net, pair: p.pair, prec: variant, ok: false, err: r.error.error }); continue; }
    const dump = (k) => { const b = E.dumps.get(`${tag}/${k}`); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
    const x = (k) => rdF32(p.files[k].file);
    const o = path.join(work, 'learned', p.net, p.pair, variant);
    fs.mkdirSync(o, { recursive: true });
    fs.writeFileSync(path.join(o, 'pf.f32'), E.dumps.get(`${tag}/pf`));
    fs.writeFileSync(path.join(o, 'est.json'), JSON.stringify({ sad: r.sad, cut: r.cut }));
    for (const t of [0.5, 0.25]) { const b = E.dumps.get(`${tag}/out_t${t}`); if (b) fs.writeFileSync(path.join(o, `out_t${t}.rgba`), b); }
    const row = { net: p.net, pair: p.pair, prec: variant, size: `${p.W}x${p.H}` };
    if (p.files.ya && !f16in) { row.ya = cmp(dump('ya'), x('ya')); row.yc = cmp(dump('yc'), x('yc')); }
    row.top = {};
    for (const k of ['in_top', 'act_top0', 'act_top1', 'act_top2', 'act_top3', 'raw_top']) row.top[k] = cmp(dump(k), x(k)).max;
    row.topMax = Math.max(...Object.values(row.top));
    row.levels = {};
    for (let l = 4; l >= 1; l--) row.levels[l] = cmp(dump('h_level' + l), x('h_level' + l));
    row.pf = cmp(dump('pf'), x('pf'));
    row.sad = r.sad; row.sadExp = p.sad; row.sadRel = Math.abs(r.sad - p.sad) / Math.max(1, p.sad);
    row.cut = r.cut; row.cutExp = p.cut; row.duplicate = r.duplicate; row.dupDiff = r.dupDiff;
    row.passCounts = r.passCounts; row.glError = r.glError;
    row.middle = p.middleWhy;
    if (p.files.pf_zt) {   // the zero-motion test vs the *_zt vectors
      const Z = r.zt, fwh = p.flow[0] * p.flow[1];
      const zpf = dump('zt_pf'), zt = dump('zt_zt'), xpf = x('pf_zt'), xmask = fs.readFileSync(p.files.zero_zt.file);
      const ch = p.files.cost_h_zt ? x('cost_h_zt') : null, c0 = p.files.cost0_zt ? x('cost0_zt') : null;
      let agree = 0, zeroed = 0, pfMax = 0, pfSum = 0, nAg = 0, chMax = 0, chAgMax = 0, c0Max = 0;
      for (let i = 0; i < fwh; i++) {
        const m = zt[4 * i + 3] > 0.5;
        if (m) zeroed++;
        const ag = m === (xmask[i] === 1);
        if (ag) {
          agree++;
          for (let k = 0; k < 2; k++) { const d = Math.abs(zpf[2 * i + k] - xpf[2 * i + k]); pfSum += d; nAg++; if (!(d <= pfMax)) pfMax = d; }
        }
        if (ch) { const d = Math.abs(zt[4 * i + 1] - ch[i]); if (!(d <= chMax)) chMax = d; if (ag && !(d <= chAgMax)) chAgMax = d; }
        if (c0) { const d = Math.abs(zt[4 * i + 2] - c0[i]); if (!(d <= c0Max)) c0Max = d; }
      }
      row.zt = { agree: agree / fwh, flipped: fwh - agree, zeroed: zeroed / fwh, zeroedExp: p.zeroed_zt, pfMax, pfMean: pfSum / nAg,
        chMax: ch ? chMax : null, chAgMax: ch ? chAgMax : null, c0Max: c0 ? c0Max : null, sad: Z.sad, sadExp: p.sad_zt,
        sadRel: Math.abs(Z.sad - p.sad_zt) / Math.max(1, p.sad_zt), cut: Z.cut, cutExp: p.cut_zt, duplicate: Z.duplicate,
        pfNetDiff: Z.pfNetDiff, passCounts: Z.passCounts };
      const oz = path.join(o, 'zt');
      fs.mkdirSync(oz, { recursive: true });
      fs.writeFileSync(path.join(oz, 'pf.f32'), E.dumps.get(`${tag}/zt_pf`));
      fs.writeFileSync(path.join(oz, 'est.json'), JSON.stringify({ sad: Z.sad, cut: Z.cut }));
      for (const t of [0.5, 0.25]) { const b = E.dumps.get(`${tag}/zt_out_t${t}`); if (b) fs.writeFileSync(path.join(oz, `out_t${t}.rgba`), b); }
    }
    rows.push(row);
  }
}
// end to end through proto/interp.py's synth
process.stdout.write(py('synth'));
const syn = JSON.parse(fs.readFileSync(path.join(work, 'learned', 'synth.json'), 'utf8'),
  (k, v) => ((k === 'synthDb' || k === 'engineDb') && v === null ? Infinity : v));   // (null: identical)
console.log('\nnet  pair    prec size       ya/yc   top max  h4       h3       h2       h1       pf mean  pf max   sad (expected)      rel      cut   ' +
  'synth dB t.5/.25  vs middle (exp)    out MAE  result');
for (const row of rows) {
  if (row.err) { console.log(`${row.net} ${row.pair} ${row.prec}: FAIL ${row.err}`); continue; }
  const s = syn[`${row.net}/${row.pair}/${row.prec}`] || {};
  row.synth = s;
  const exact = row.prec === 'fp32', fails = [];
  if (row.ya && (row.ya.neq || row.yc.neq)) fails.push(`ya/yc ${row.ya.neq}+${row.yc.neq} px differ (max ${row.ya.max}, ${row.yc.max})`);
  if (exact) {   // the exact float32 port: SPEC section 8's tolerances
    if (!(row.topMax <= TOL.top)) fails.push('top ' + e1(row.topMax));
    if (!(row.pf.mean <= TOL.pfMean)) fails.push('pf mean ' + e1(row.pf.mean));
    if (!(row.pf.max <= TOL.pfMax)) fails.push('pf max ' + e1(row.pf.max));
  }
  if (!(row.sadRel <= (exact ? TOL.sadRel : TOL.sadRel16))) fails.push('sad ' + e1(row.sadRel));
  if (row.cut !== row.cutExp) fails.push('cut');
  if (row.duplicate !== false) fails.push('duplicate ' + row.duplicate);
  if (row.glError) fails.push('glError ' + row.glError);
  for (const t of ['0.5', '0.25']) {
    const v = s[t];
    if (!v) { fails.push('no synth t=' + t); continue; }
    if (!(v.synthDb >= TOL.synthDb)) fails.push(`synth t=${t} ${v.synthDb.toFixed(1)} dB`);
    if (!(v.engineMae <= TOL.engineMae)) fails.push(`engine out t=${t} MAE ${v.engineMae}`);
    if (v.midDelta !== undefined && !(Math.abs(v.midDelta) <= TOL.midDb)) fails.push(`vs middle frame ${v.midDelta.toFixed(3)} dB`);
  }
  row.ok = fails.length === 0;
  row.fails = fails;
  if (!row.ok) allOk = false;
  const db = (t) => (s[t] ? (Number.isFinite(s[t].synthDb) ? s[t].synthDb.toFixed(1) : 'inf') : '-');
  const mid = s['0.5'] && s['0.5'].midDb !== undefined ? `${s['0.5'].midDb.toFixed(3)} (${s['0.5'].midDbExp.toFixed(3)})` : '-';
  const mae = Math.max(...['0.5', '0.25'].map((t) => (s[t] ? s[t].engineMae : NaN)));
  console.log(`${row.net} ${row.pair.padEnd(7)} ${row.prec} ${row.size.padEnd(10)} ${(row.ya ? (row.ya.neq + row.yc.neq === 0 ? 'exact' : 'DIFF') : '-').padEnd(7)} ${e1(row.topMax).padEnd(8)} ` +
    [4, 3, 2, 1].map((l) => e1(row.levels[l].max).padEnd(8)).join(' ') + ` ${e1(row.pf.mean).padEnd(8)} ${e1(row.pf.max).padEnd(8)} ` +
    `${row.sad.toFixed(4)} (${row.sadExp.toFixed(4)})`.padEnd(19) + ` ${e1(row.sadRel).padEnd(8)} ${String(row.cut).padEnd(5)} ${(db('0.5') + ' / ' + db('0.25')).padEnd(16)} ` +
    `${mid.padEnd(18)} ${mae.toFixed(3).padEnd(8)} ` + (row.ok ? 'PASS' : 'FAIL ' + fails.join('; ')));
}

// the zero-motion test rows
console.log('\nzero-motion test (learnedZeroTest true) vs the *_zt vectors:\nnet  pair    prec mask agree   flipped zeroed (exp)    pf mean  pf max   C(h) max C(0) max ' +
  'sad_zt (expected)   rel      cut   synth dB t.5/.25  vs middle (exp)    out MAE  result');
for (const row of rows) {
  if (row.err || !row.zt) continue;
  const z = row.zt, s = syn[`${row.net}/${row.pair}/${row.prec}/zt`] || {};
  z.synth = s;
  const exact = row.prec === 'fp32', fails = [];
  if (exact) {
    if (!(z.agree >= TOL.ztAgree)) fails.push('mask agreement ' + (100 * z.agree).toFixed(4) + '%');
    if (!(z.pfMean <= TOL.pfMean)) fails.push('pf mean ' + e1(z.pfMean));
    if (!(z.pfMax <= TOL.pfMax)) fails.push('pf max ' + e1(z.pfMax));
    if (z.c0Max !== null && row.ya && !(z.c0Max <= TOL.cost)) fails.push('C(0) ' + e1(z.c0Max));
  }
  if (!(z.sadRel <= (exact ? TOL.sadRel : TOL.sadRel16))) fails.push('sad_zt ' + e1(z.sadRel));
  if (z.cut !== z.cutExp) fails.push('cut');
  if (z.duplicate !== false) fails.push('duplicate ' + z.duplicate);
  if (z.pfNetDiff !== 0) fails.push(`net field differs from the run without the test (${z.pfNetDiff} values)`);
  for (const t of ['0.5', '0.25']) {
    const v = s[t];
    if (!v) { fails.push('no synth t=' + t); continue; }
    if (!(v.synthDb >= (exact ? TOL.ztSynthDb : TOL.synthDb))) fails.push(`synth t=${t} ${v.synthDb.toFixed(1)} dB`);
    if (!(v.engineMae <= TOL.engineMae)) fails.push(`engine out t=${t} MAE ${v.engineMae}`);
    if (v.midDelta !== undefined && !(Math.abs(v.midDelta) <= TOL.midDb)) fails.push(`vs middle frame ${v.midDelta.toFixed(3)} dB`);
  }
  z.ok = fails.length === 0;
  z.fails = fails;
  if (!z.ok) allOk = false;
  const db = (t) => (s[t] ? (Number.isFinite(s[t].synthDb) ? s[t].synthDb.toFixed(1) : 'inf') : '-');
  const mid = s['0.5'] && s['0.5'].midDb !== undefined ? `${s['0.5'].midDb.toFixed(3)} (${s['0.5'].midDbExp.toFixed(3)})` : '-';
  const mae = Math.max(...['0.5', '0.25'].map((t) => (s[t] ? s[t].engineMae : NaN)));
  const f = (v) => (v === null ? '-' : e1(v));
  console.log(`${row.net} ${row.pair.padEnd(7)} ${row.prec} ${(100 * z.agree).toFixed(4).padStart(8)}%  ${String(z.flipped).padEnd(7)} ` +
    `${(100 * z.zeroed).toFixed(1)}% (${(100 * z.zeroedExp).toFixed(1)}%)`.padEnd(15) + ` ${e1(z.pfMean).padEnd(8)} ${e1(z.pfMax).padEnd(8)} ` +
    `${f(z.chMax).padEnd(8)} ${f(z.c0Max).padEnd(8)} ` + `${z.sad.toFixed(4)} (${z.sadExp.toFixed(4)})`.padEnd(19) + ` ${e1(z.sadRel).padEnd(8)} ` +
    `${String(z.cut).padEnd(5)} ${(db('0.5') + ' / ' + db('0.25')).padEnd(16)} ${mid.padEnd(18)} ${mae.toFixed(3).padEnd(8)} ` + (z.ok ? 'PASS' : 'FAIL ' + fails.join('; ')));
}

// the duplicate skip, per net, precision and zero test on / off, on the motion pair's first frame
const dups = {};
for (const prec of opt.prec) for (const zt of [true, false]) {
  for (const net of [...new Set(man.pairs.map((q) => q.net))]) {
    if (opt.nets && !opt.nets.includes(net)) continue;
    const p = man.pairs.find((q) => q.net === net && q.pair === 'motion');
    if (!p) continue;
    const r = await E.page.evaluate((j) => window.__dupTest(j), { net, prec, zt, dir: `learned/${net}/${p.pair}`, W: p.W, H: p.H });
    const dn = `${net} ${prec} zero test ${zt ? 'on' : 'off'}`;
    dups[net + '/' + prec + '/' + (zt ? 'zt' : 'net')] = r;
    if (r.error) { console.log(`dup ${dn}: ENGINE ERROR ${r.error.error}\n${r.error.stack}`); allOk = false; continue; }
    const I = r.identical, N = r.near;
    const okI = I.estimate && I.duplicate === true && I.dupDiff === 0 && I.nonZeroField === 0 && I.meanSad === 0 && !I.cut &&
      I['synth0.5'] && I['synth0.25'] && I['diffVsA0.5'] === 0 && I['diffVsA0.25'] === 0;
    const okN = N.duplicate === true && N.dupDiff > 0 && N.dupDiff < 0.6 && N.nonZeroField === 0 && N.meanSad === 0;
    const ok = okI && okN && !r.glError;
    if (!ok) allOk = false;
    console.log(`dup ${dn}: identical frames -> duplicate ${I.duplicate} (mean |a-c| ${I.dupDiff}), non-zero field components ${I.nonZeroField}, ` +
      `sad ${I.meanSad}, output vs input: ${I['diffVsA0.5']} / ${I['diffVsA0.25']} channel values differ (t 0.5 / 0.25); near-duplicate (mean |a-c| ` +
      `${N.dupDiff.toFixed(3)}) -> duplicate ${N.duplicate}, non-zero field ${N.nonZeroField}, sad ${N.meanSad}: ${ok ? 'PASS' : 'FAIL'}`);
  }
}
// the static guard vs static_guard.py
const sgRows = [];
{
  const sgo = path.join(work, 'learned', 'sguard');
  fs.rmSync(sgo, { recursive: true, force: true });
  const cutPair = (man.pairs.find((q) => q.net === SG_NET && q.cut_zt) || {}).pair;   // a fallback frame (vec.json's cut_zt)
  const cases = [['overlay', false, [0.5, 0.25]], ['motion', false, [0.25]], [cutPair, false, [0.5]], ['overlay', true, [0.25]]];
  if (!cutPair) { console.log(`static guard: no ${SG_NET} pair with cut_zt: FAIL`); allOk = false; }
  for (const [pair, f16in, ts] of cases) {
    const p = man.pairs.find((q) => q.net === SG_NET && q.pair === pair);
    if (!p || (opt.nets && !opt.nets.includes(SG_NET)) || !opt.prec.includes('fp32')) continue;
    const tag = `sg_${pair}${f16in ? '_hdr' : ''}`;
    const r = await E.page.evaluate((j) => window.__sgTest(j), { net: SG_NET, prec: 'fp32', f16in, ts, dir: `learned/${SG_NET}/${pair}`, W: p.W, H: p.H, tag });
    if (r.error) { console.log(`${tag}: ENGINE ERROR ${r.error.error}\n${r.error.stack}`); allOk = false; continue; }
    for (const t of ts) {
      const name = `${pair}${f16in ? '_hdr' : ''}_t${t}`, d = path.join(sgo, name), rt = r.t[t];
      fs.mkdirSync(d, { recursive: true });
      for (const k of ['a', 'c']) fs.copyFileSync(path.join(work, 'learned', SG_NET, pair, k + '.rgba'), path.join(d, k + '.rgba'));
      for (const k of ['out0', 'outg']) fs.writeFileSync(path.join(d, k + (f16in ? '.f32' : '.rgba')), E.dumps.get(`${tag}/${t}/${k}`));
      fs.writeFileSync(path.join(d, 'field.f32'), E.dumps.get(`${tag}/${t}/field`));
      fs.writeFileSync(path.join(d, 'meta.json'), JSON.stringify({ W: p.W, H: p.H, fw: p.flow[0], fh: p.flow[1], t, flowScale: 2, f16: f16in, fallback: rt.fallback }));
      sgRows.push({ name, ...rt, glError: r.glError });
    }
  }
  if (sgRows.length) {
    process.stdout.write(py('sguard'));
    const sgj = JSON.parse(fs.readFileSync(path.join(work, 'learned', 'sguard.json'), 'utf8'));
    for (const row of sgRows) {
      const s = sgj[row.name], fails = [];
      row.py = s;
      if (!row.ok) fails.push('synth returned false');
      if (!s) fails.push('no static_guard.py result');
      else if (!(s.maxDiff <= TOL.sgMax)) fails.push(`max |d| ${s.maxDiff.toFixed(3)} levels`);
      if (row.offDiff !== 0) fails.push(`guard off again: ${row.offDiff} values differ from the unguarded output`);
      if (row.fallback && row.guardDiff !== 0) fails.push(`fallback ${row.fallback} changed by the guard (${row.guardDiff} values)`);
      if (!row.fallback && row.guardDiff === 0) fails.push('the guard changed nothing');
      if (row.glError) fails.push('glError ' + row.glError);
      row.ok = fails.length === 0;
      if (!row.ok) allOk = false;
      console.log(`static guard ${row.name}: fallback ${row.fallback}, guard changed ${row.guardDiff} channel values, guard off again: ${row.offDiff} differ` +
        (s ? `, vs static_guard.py max |d| ${s.maxDiff.toFixed(3)} levels (${s.nDiff} px differ)` : '') + `: ${row.ok ? 'PASS' : 'FAIL ' + fails.join('; ')}`);
    }
  }
}
if (E.consoleMsgs.size) console.log('console: ' + [...E.consoleMsgs.entries()].map(([k, n]) => `${k} (x${n})`).join('\n         '));
if (opt.json) fs.writeFileSync(opt.json, JSON.stringify({ browser: E.version, renderer: E.init.info.renderer, gpu: opt.gpu, rows, dups, sguard: sgRows }, null, 1));
console.log(`RESULT: ${allOk ? 'PASS' : 'FAIL'}`);
await E.browser.close();
process.exit(allOk ? 0 : 1);
