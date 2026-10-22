// Metrics + stage specs. Arrays are {w,h,c,data} row-major (H,W,C), row 0 = top.

export const DIFFER_EPS = 1e-3;   // a vector "differs" if its endpoint error exceeds this (px)

// NaN / Inf in the engine output counts as an infinite error (never silently 0)
const fin = (d) => (Number.isFinite(d) ? d : Infinity);

function worstIdx(w, i) { return { x: i % w, y: Math.floor(i / w) }; }

/** component-wise scalar compare of channel ch (or all RGB if ch==='rgb') */
export function scalar(e, r, ch = 0, tol = DIFFER_EPS) {
  const n = e.w * e.h;
  let sum = 0, max = 0, mi = 0, bad = 0;
  for (let i = 0; i < n; i++) {
    const d = fin(Math.abs(e.data[i * e.c + ch] - r.data[i * r.c + ch]));
    sum += d; if (d > max) { max = d; mi = i; } if (d > tol) bad++;
  }
  const { x, y } = worstIdx(e.w, mi);
  return { mean: sum / n, max, pctDiffer: 100 * bad / n, worst: { x, y, eng: e.data[mi * e.c + ch], ref: r.data[mi * r.c + ch] }, n };
}

/** 2-vector endpoint error over channels 0,1 */
export function vec2(e, r, eps = DIFFER_EPS) {
  const n = e.w * e.h;
  let sum = 0, max = 0, mi = 0, bad = 0;
  for (let i = 0; i < n; i++) {
    const dx = e.data[i * e.c] - r.data[i * r.c], dy = e.data[i * e.c + 1] - r.data[i * r.c + 1];
    const d = fin(Math.hypot(dx, dy));
    sum += d; if (d > max) { max = d; mi = i; } if (d > eps) bad++;
  }
  const { x, y } = worstIdx(e.w, mi);
  const g = (a, i) => [a.data[i * a.c], a.data[i * a.c + 1]].map((v) => +v.toFixed(4));
  return { mean: sum / n, max, pctDiffer: 100 * bad / n, worst: { x, y, eng: g(e, mi), ref: g(r, mi) }, n };
}

/** 8-bit RGB images (RGBA, alpha ignored): MAE over RGB (levels), max abs, % pixels whose worst channel is >4 off */
export function rgb(e, r) {
  const n = e.w * e.h;
  let sum = 0, max = 0, mi = 0, bad = 0, big = 0;
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let k = 0; k < 3; k++) {
      const d = Math.abs(e.data[i * e.c + k] - r.data[i * r.c + k]);
      sum += d; if (d > m) m = d;
    }
    if (m > max) { max = m; mi = i; }
    if (m > 4) big++;
    if (m > 0) bad++;
  }
  const { x, y } = worstIdx(e.w, mi);
  const g = (a, i) => [0, 1, 2].map((k) => a.data[i * a.c + k]);
  return { mean: sum / (3 * n), max, pct4: 100 * big / n, pctDiffer: 100 * bad / n, worst: { x, y, eng: g(e, mi), ref: g(r, mi) }, n };
}

// ---------------------------------------------------------------------------- stage specs
// kind: how to compare; ok(m): "not diverged" predicate (informational, for 'first diverged stage');
// only `out` + decisions gate the PASS/FAIL (D11).
const lumaOk = (m) => m.max <= 1e-3;
const fieldOk = (m) => m.pctDiffer <= 2;
const imgOk = (m) => m.mean <= 0.5;

/** returns [{stage, kind, name, ch?, ok, note}] in pipeline order. Names are debugRead keys. */
export function pairStages(levels) {
  const L = levels.length;
  const s = [{ stage: 'frame.prev', name: 'frame.prev', kind: 'rgb', ok: (m) => m.max === 0, note: 'upload / orientation (must be exact)' },
    { stage: 'frame.curr', name: 'frame.curr', kind: 'rgb', ok: (m) => m.max === 0, note: 'upload / orientation (must be exact)' }];
  for (let i = 0; i < L; i++) for (const w of ['prev', 'curr']) s.push({ stage: `luma.${w}.${i}`, name: `luma.${w}.${i}`, kind: 'scalar', ok: lumaOk, note: 'max<=1e-3' });
  for (let i = L - 1; i >= 0; i--) {
    if (i < L - 1) {
      s.push({ stage: `median.${i}`, name: `median.${i}`, kind: 'vec2', ok: (m) => m.max <= 0.01, note: 'global median (2*m)' });
      s.push({ stage: `seed.${i}`, name: `seed.${i}`, kind: 'vec2', ok: fieldOk, note: '<=2% blocks differ' });
    }
    s.push({ stage: `field.${i}`, name: `field.${i}`, kind: 'vec2', ok: fieldOk, note: i === 0 ? '<=2% blocks differ (integer, pre-subpixel)' : '<=2% blocks differ' });
  }
  s.push({ stage: 'sub1.xy', name: 'sub1', kind: 'vec2', ok: fieldOk, note: '0.5px pass; <=2% blocks differ' });
  s.push({ stage: 'sub1.z', name: 'sub1', kind: 'scalar', ch: 2, ok: (m) => m.mean <= 0.02, note: 'block mean SAD; mean<=0.02' });
  s.push({ stage: 'sub2.xy', name: 'sub2', kind: 'vec2', ok: fieldOk, note: 'final block vectors; <=2% blocks differ' });
  s.push({ stage: 'sub2.z', name: 'sub2', kind: 'scalar', ch: 2, ok: (m) => m.mean <= 0.02, note: 'block mean SAD; mean<=0.02' });
  s.push({ stage: 'sadSum', name: 'sadSum', kind: 'scalar', rel: true, ok: (m) => m.max <= 1e-3 * Math.max(1, Math.abs(m.worst.ref)), note: 'rel<=1e-3' });
  s.push({ stage: 'pixfield', name: 'pixfield', kind: 'vec2', ok: fieldOk, note: '<=2% px differ' });
  return s;
}

export function tStages(t) {
  const s = [];
  if (t !== 0.5) s.push({ stage: 'fieldT', name: 'fieldT', kind: 'vec2', ok: (m) => m.mean <= 0.01, note: 'mean<=0.01 px' });
  s.push({ stage: 'warpA', name: 'warpA', kind: 'rgb', ok: imgOk, note: 'MAE<=0.5' });
  s.push({ stage: 'warpC', name: 'warpC', kind: 'rgb', ok: imgOk, note: 'MAE<=0.5' });
  s.push({ stage: 'mismatch', name: 'mismatch', kind: 'scalar', ok: (m) => m.mean <= 0.5, note: 'mean<=0.5 level' });
  s.push({ stage: 'trust', name: 'trust', kind: 'scalar', ok: (m) => m.mean <= 0.01, note: 'mean<=0.01' });
  s.push({ stage: 'untrusted', name: 'untrusted', kind: 'scalar', ok: (m, W, H) => m.max <= 0.002 * W * H, note: 'count within 0.2% of W*H' });
  s.push({ stage: 'out', name: 'out', kind: 'rgb', ok: (m) => m.mean <= 1.0, note: 'MAE<=1 level (D11 gate)' });
  return s;
}

export function compareStage(spec, e, r) {
  if (e.error) return { status: 'ERROR', error: e.error };
  if (e.width !== r.w || e.height !== r.h || e.channels !== r.c) {
    return { status: 'SHAPE', error: `shape mismatch: engine ${e.width}x${e.height}x${e.channels} vs reference ${r.w}x${r.h}x${r.c}` };
  }
  const ea = { w: e.width, h: e.height, c: e.channels, data: e.data };
  let m;
  if (spec.kind === 'rgb') m = rgb(ea, r);
  else if (spec.kind === 'vec2') m = vec2(ea, r);
  else m = scalar(ea, r, spec.ch || 0, spec.rel ? 0 : DIFFER_EPS);
  return { status: 'CMP', m };
}
