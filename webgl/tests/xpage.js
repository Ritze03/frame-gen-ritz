// Runs inside the test page (http://fg.test/, served by xbackend.mjs / bench.mjs through Playwright routes) after framegen.js attached
// globalThis.__fg. Exposes window.__x: init, run (one pair x settings: push, estimate, synth per t, compare with the Vulkan reference
// in the page, return small metrics), switchTest, bench. Files are fetched from /w/<path relative to $FG_WORK>.
(() => {
  let gl = null, engine = null, srcs = null;
  const err = (e) => ({ error: String((e && e.message) || e), stack: e && e.stack ? String(e.stack).split('\n').slice(0, 8).join('\n') : '' });
  const get = async (p) => { const r = await fetch('/w/' + p); if (!r.ok) throw new Error('fetch ' + p + ': ' + r.status); return r.arrayBuffer(); };
  const getOpt = async (p) => { const r = await fetch('/w/' + p); return r.ok ? r.arrayBuffer() : null; };

  // the library's debug maps (gpu/fgmaps.py): 16-word header, block field bw*bh vec2, per-pixel field fw*fh vec2, e part
  function parseMaps(buf) {
    const u = new Uint32Array(buf, 0, 16);
    if (u[0] !== 0x504d4746) throw new Error('fgmaps: bad magic');
    const [W, H, fw, fh, bw, bh, fscale, eStride] = Array.from(u.slice(2, 10));
    const block = new Float32Array(buf, 64, bw * bh * 2);
    const pix = new Float32Array(buf, 64 + bw * bh * 8, fw * fh * 2);
    const eOff = 64 + bw * bh * 8 + fw * fh * 8;
    const eWords = new Uint32Array(buf, eOff, eStride * H);
    return { W, H, fw, fh, bw, bh, fscale, eStride, block, pix, eWords, motionVersion: u[14], v2Mode: u[15] };
  }
  // endpoint error of an engine 2-vector texture (channels c) vs a packed vec2 array
  function epe(eng, ref) {
    const n = eng.width * eng.height, c = eng.channels;
    let s = 0, mx = 0, bad = 0;
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(eng.data[i * c] - ref[2 * i], eng.data[i * c + 1] - ref[2 * i + 1]);
      s += d; if (d > mx) mx = d; if (d > 1e-3) bad++;
    }
    return { mean: s / n, max: mx, pctDiffer: 100 * bad / n };
  }
  // RGB(A8) images: MAE over RGB, max, % pixels whose worst channel is > 4 off, % differing, PSNR
  function rgbCmp(e, r, n) {
    let sum = 0, sq = 0, mx = 0, big = 0, bad = 0;
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let k = 0; k < 3; k++) { const d = Math.abs(e[i * 4 + k] - r[i * 4 + k]); sum += d; sq += d * d; if (d > m) m = d; }
      if (m > mx) mx = m;
      if (m > 4) big++;
      if (m > 0) bad++;
    }
    const mse = sq / (3 * n);
    return { mae: sum / (3 * n), max: mx, pct4: 100 * big / n, pctDiffer: 100 * bad / n, psnr: mse === 0 ? Infinity : 10 * Math.log10(255 * 255 / mse) };
  }

  window.__x = {
    init({ sources, settings }) {
      srcs = sources;
      const canvas = new OffscreenCanvas(64, 64);
      gl = canvas.getContext('webgl2', { antialias: false, alpha: true, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: 'high-performance' });
      if (!gl) return { fatal: 'getContext("webgl2") returned null' };
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      const info = { renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER), maxTex: gl.getParameter(gl.MAX_TEXTURE_SIZE) };
      try { info.support = __fg.checkSupport(gl); engine = __fg.createEngine(gl, sources, settings); } catch (e) { return { fatal: err(e).error, info }; }
      return { info };
    },
    setSettings(p) { try { return engine.setSettings(p); } catch (e) { return err(e); } },
    _gl() { return gl; },
    _engine() { return engine; },

    // job: { pair, W, H, settings, ts, ref: 'vkref/<cfg>/<pair>', mode }
    async run(job) {
      const R = { t: {}, calls: {} };
      try {
        const a = new Uint8Array(await get(`pairs/${job.pair}/a.rgba`)), c = new Uint8Array(await get(`pairs/${job.pair}/c.rgba`));
        R.calls.setSettings = engine.setSettings(job.settings);
        const t0 = performance.now();
        engine.pushFrame({ width: job.W, height: job.H, data: a });
        engine.pushFrame({ width: job.W, height: job.H, data: c });
        R.calls.estimate = engine.estimate();
        const n = job.W * job.H;
        let first = true;
        for (const t of job.ts) {
          const T = R.t[t] = {};
          T.synth = engine.synth(t, undefined);
          if (!T.synth) continue;
          T.stats = engine.stats();
          const out = engine.debugRead('out');
          if (first) { gl.finish(); R.ms = performance.now() - t0; first = false; }
          const refBuf = await getOpt(`${job.ref}/t${t}.rgba`);
          if (!refBuf) { T.noRef = true; continue; }
          const ref = new Uint8Array(refBuf);
          T.cmp = rgbCmp(out.data, ref, n);
          T.libStats = new TextDecoder().decode(await get(`${job.ref}/t${t}.stats`)).trim().split(/\s+/).slice(2).map(Number);
          if (job.dump) {
            await fetch(`/dump/${job.dumpName}_t${t}`, { method: 'POST', body: out.data });
          }
          // stage diagnostics vs the library's debug maps
          const maps = parseMaps(await get(`${job.ref}/t${t}.fgmaps`));
          const st = T.stages = {};
          if (job.mode === 'v1') {
            st.blockField = epe(engine.debugRead('sub2'), maps.block);
            st.pixField = epe(engine.debugRead('pixfield'), maps.pix);
          } else {
            st.blockField = epe(engine.debugRead('v2.vf'), maps.block);
            if (job.mode === 'v2q') {
              st.pixField = epe(engine.debugRead('v2.qf'), maps.pix);
              if (maps.fscale >= 2 && maps.eStride * maps.H >= maps.fw * maps.fh) {   // (the library dumps the coverage only if it fits the e part)
                const cov = engine.debugRead('v2.qcov');
                let s = [0, 0, 0], mx = [0, 0, 0], k = maps.fw * maps.fh;
                for (let i = 0; i < k; i++) {
                  const w = maps.eWords[i];
                  const r3 = [(w & 2047) / 2047, ((w >>> 11) & 2047) / 2047, (w >>> 22) / 1023];
                  for (let j = 0; j < 3; j++) { const d = Math.abs(cov.data[i * 4 + j] - r3[j]); s[j] += d; if (d > mx[j]) mx[j] = d; }
                }
                st.coverage = { meanP: s[0] / k, meanC: s[1] / k, meanZ: s[2] / k, maxP: mx[0], maxC: mx[1], maxZ: mx[2] };
              }
            }
          }
        }
        R.passCounts = engine.passCounts();
        R.glError = gl.getError();
      } catch (e) { R.error = err(e); }
      return R;
    },

    // API checks: size limits reported cleanly, settings validation, a refused resize keeps the engine usable
    apiTest() {
      const r = {};
      const throws = (f) => { try { f(); return false; } catch (e) { return String(e.message || e); } };
      const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      r.tooSmall = __fg.frameSizeError(gl, 8, 8, {});
      r.minOk = __fg.frameSizeError(gl, 16, 16, {});
      r.tooBig = __fg.frameSizeError(gl, max + 1, 64, {});
      r.wide = __fg.frameSizeError(gl, max, 64, { flowScale: 1 });
      r.support = __fg.checkSupport(gl, { width: max + 1, height: 64 });
      r.supportFits = __fg.checkSupport(gl, { width: 64, height: 48 });
      r.badMode = throws(() => engine.setSettings({ v2Mode: 'fast' }));
      r.badFormat = throws(() => engine.setSettings({ inputFormat: 'rgb565' }));
      r.badBlur = throws(() => engine.setSettings({ blurSamples: 17 }));
      engine.setSettings({ motionVersion: 1, flowScale: 2 });
      engine.pushFrame({ width: 64, height: 48, data: new Uint8Array(64 * 48 * 4) });
      r.bigResize = throws(() => engine.resize(max + 1, 64));
      r.infoAfter = engine.info && [engine.info.W, engine.info.H];
      r.v1Blur = throws(() => { engine.setSettings({ blurSamples: 4 }); engine.pushFrame({ width: 64, height: 48, data: new Uint8Array(64 * 48 * 4) }); engine.estimate(); engine.synthBlur(0.25, 0.75); });
      engine.setSettings({ blurSamples: 0 });
      r.version = __fg.version;
      r.glError = gl.getError();
      return r;
    },

    // checkSupport's MIN-blend probe (V2 Quality): its result on this context, cached, GL state left as it was; then a context whose
    // MIN blend is broken (a Proxy that turns blendEquation(MIN) into FUNC_ADD): quality false, v2Mode 'quality' refused with a
    // RangeError (createEngine and setSettings, settings kept), V1 and V2 Performance still run
    qualityTest() {
      const r = {};
      const throws = (f) => { try { f(); return false; } catch (e) { return (e && e.name) + ': ' + String(e.message || e); } };
      try {
        r.real = __fg.checkSupport(gl);
        r.cached = __fg.checkSupport(gl).qualityError === r.real.qualityError;
        const g2 = new OffscreenCanvas(16, 16).getContext('webgl2');
        g2.viewport(1, 2, 3, 4); g2.enable(g2.BLEND); g2.blendEquationSeparate(g2.FUNC_SUBTRACT, g2.FUNC_REVERSE_SUBTRACT);
        g2.clearColor(0.25, 0.5, 0.75, 1); g2.colorMask(true, false, true, false); g2.enable(g2.SCISSOR_TEST);
        g2.activeTexture(g2.TEXTURE3);
        const fb = g2.createFramebuffer(); g2.bindFramebuffer(g2.FRAMEBUFFER, fb);
        const t0 = performance.now();
        r.fresh = __fg.checkSupport(g2);
        r.probeMs = performance.now() - t0;
        const P = (k) => Array.from([].concat(g2.getParameter(k)));
        r.stateKept = g2.getParameter(g2.FRAMEBUFFER_BINDING) === fb && P(g2.VIEWPORT).join() === '1,2,3,4' && g2.isEnabled(g2.BLEND) &&
          g2.getParameter(g2.BLEND_EQUATION_RGB) === g2.FUNC_SUBTRACT && g2.getParameter(g2.BLEND_EQUATION_ALPHA) === g2.FUNC_REVERSE_SUBTRACT &&
          P(g2.COLOR_CLEAR_VALUE).join() === '0.25,0.5,0.75,1' && P(g2.COLOR_WRITEMASK).join() === 'true,false,true,false' &&
          g2.isEnabled(g2.SCISSOR_TEST) && g2.getParameter(g2.ACTIVE_TEXTURE) === g2.TEXTURE3 && g2.getError() === g2.NO_ERROR;
        g2.bindFramebuffer(g2.FRAMEBUFFER, null);
        const broken = new Proxy(g2, { get(t, k) {
          if (k === 'blendEquation') return (m) => t.blendEquation(m === t.MIN ? t.FUNC_ADD : m);
          const v = t[k]; return typeof v === 'function' ? v.bind(t) : v;
        } });
        r.broken = __fg.checkSupport(broken);
        r.createQ = throws(() => __fg.createEngine(broken, srcs, { motionVersion: 2, v2Mode: 'quality' }));
        const e = __fg.createEngine(broken, srcs, { motionVersion: 2, v2Mode: 'performance' });
        const fr = (k) => { const d = new Uint8Array(64 * 48 * 4); for (let i = 0; i < d.length; i++) d[i] = ((i >> 2) * 7 + k * 3 + (i & 3) * 50) & 255; return { width: 64, height: 48, data: d }; };
        e.pushFrame(fr(0)); e.pushFrame(fr(1));
        r.perf = [e.estimate(), e.synth(0.5, undefined)];
        r.setQ = throws(() => e.setSettings({ v2Mode: 'quality' }));
        r.keptMode = e.settings.v2Mode + '/' + e.settings.motionVersion;
        e.setSettings({ motionVersion: 1, v2Mode: 'quality' });   // V1 never needs it (and 'quality' is the default v2Mode)
        r.v1 = [e.estimate(), e.synth(0.5, undefined)];
        r.toV2 = throws(() => e.setSettings({ motionVersion: 2 }));
        r.engineSupport = e.support && e.support.quality;
        e.dispose();
        r.glError = g2.getError();
        const lose = g2.getExtension('WEBGL_lose_context'); if (lose) lose.loseContext();
      } catch (e) { r.error = err(e); }
      return r;
    },

    // motion blur job: { pair, W, H, settings (with blurSamples, blurWeights), windows: [{tag, t0, t1, ref}] } -> per window the
    // synthBlur output vs the library's recordSynthBlur (rgba8)
    async runBlur(job) {
      const R = { w: {} };
      try {
        const a = new Uint8Array(await get(`pairs/${job.pair}/a.rgba`)), c = new Uint8Array(await get(`pairs/${job.pair}/c.rgba`));
        engine.setSettings(job.settings);
        engine.pushFrame({ width: job.W, height: job.H, data: a });
        engine.pushFrame({ width: job.W, height: job.H, data: c });
        engine.estimate();
        for (const w of job.windows) {
          const T = R.w[w.tag] = {};
          engine.setSettings({ blurSamples: w.n, blurWeights: w.weights });   // (runtime keys: the estimate stays)
          T.ok = engine.synthBlur(w.t0, w.t1, undefined);
          const out = engine.debugRead('out').data;
          const rb = await getOpt(w.ref);
          if (rb) T.cmp = rgbCmp(out, new Uint8Array(rb), job.W * job.H);
        }
        engine.setSettings({ blurSamples: 0 });
        R.glError = gl.getError();
      } catch (e) { R.error = err(e); }
      return R;
    },

    // HDR job: { dir (under $FG_WORK, with a.f16 / c.f16), W, H, settings (inputFormat rgba16f, transfer, ...), ts, refs: {t: {file,
    // fmt: 'f16' | 'rgba8', stats}} } -> per t: the output compared in levels of the perceptual encoding (linear: PQ x 255; encoded:
    // x 255; an rgba8 reference: its levels), decisions, finiteness
    async runHdr(job) {
      const R = { t: {} };
      const half = (h) => {
        const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
        return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
      };
      const white = job.settings.linearWhiteNits || 80;
      const pq = (v) => {
        const y = Math.min(Math.max(v * white / 10000, 0), 1), p = y ** 0.1593017578125;
        return ((0.8359375 + 18.8515625 * p) / (1 + 18.6875 * p)) ** 78.84375;
      };
      const lv = job.settings.transfer === 'linear' ? (v) => 255 * pq(v) : (v) => 255 * v;
      try {
        const a = new Uint16Array(await get(`${job.dir}/a.f16`)), c = new Uint16Array(await get(`${job.dir}/c.f16`));
        R.setSettings = engine.setSettings(job.settings);
        engine.pushFrame({ width: job.W, height: job.H, data: a });
        engine.pushFrame({ width: job.W, height: job.H, data: c });
        R.estimate = engine.estimate();
        const n = job.W * job.H;
        for (const t of job.ts) {
          const T = R.t[t] = {};
          T.synth = engine.synth(t, undefined);
          if (!T.synth) continue;
          T.stats = engine.stats();
          const out = engine.debugRead('out').data;
          let nonFinite = 0, over1 = 0;
          for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { const v = out[i * 4 + k]; if (!Number.isFinite(v)) nonFinite++; else if (v > 1) over1++; }
          T.nonFinite = nonFinite; T.pctOver1 = 100 * over1 / (3 * n);
          const ref = job.refs && job.refs[t];
          if (!ref) continue;
          const rb = await getOpt(ref.file);
          if (!rb) { T.noRef = true; continue; }
          const rv = ref.fmt === 'f16' ? new Uint16Array(rb) : new Uint8Array(rb);
          const rlv = ref.fmt === 'f16' ? (i) => lv(half(rv[i])) : (i) => rv[i];
          let sum = 0, mx = 0, big = 0;
          for (let i = 0; i < n; i++) {
            let m = 0;
            for (let k = 0; k < 3; k++) { const d = Math.abs(lv(out[i * 4 + k]) - rlv(i * 4 + k)); sum += d; if (d > m) m = d; }
            if (m > mx) mx = m;
            if (m > 4) big++;
          }
          T.cmp = { mae: sum / (3 * n), max: mx, pct4: 100 * big / n };
          if (ref.stats) T.libStats = new TextDecoder().decode(await get(ref.stats)).trim().split(/\s+/).slice(2).map(Number);
        }
        R.glError = gl.getError();
      } catch (e) { R.error = err(e); }
      return R;
    },

    // a motionVersion switch drops the estimate; a v2Mode switch keeps it (and gives that mode's own output)
    async switchTest({ pair, W, H, t }) {
      const r = { steps: [] };
      try {
        const a = new Uint8Array(await get(`pairs/${pair}/a.rgba`)), c = new Uint8Array(await get(`pairs/${pair}/c.rgba`));
        const px = () => engine.debugRead('out').data.slice();
        const maxd = (x, y) => { let m = 0; for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i] - y[i])); return m; };
        engine.setSettings({ motionVersion: 2, v2Mode: 'quality' });
        engine.pushFrame({ width: W, height: H, data: a });
        engine.pushFrame({ width: W, height: H, data: c });
        r.steps.push(['estQ', engine.estimate()]);
        r.steps.push(['synthQ', engine.synth(t, undefined)]);
        const q1 = px();
        engine.setSettings({ v2Mode: 'performance' });
        r.steps.push(['synthP-kept', engine.synth(t, undefined)]);
        const p1 = px();
        r.steps.push(['estP', engine.estimate()]);
        r.steps.push(['synthP', engine.synth(t, undefined)]);
        const p2 = px();
        engine.setSettings({ v2Mode: 'quality' });
        r.steps.push(['synthQ-kept', engine.synth(t, undefined)]);   // Quality's tail derived lazily from a Performance estimate
        const q2 = px();
        engine.setSettings({ motionVersion: 1 });
        r.steps.push(['synth1-before-estimate', engine.synth(t, undefined)]);
        r.steps.push(['est1', engine.estimate()]);
        r.steps.push(['synth1', engine.synth(t, undefined)]);
        engine.setSettings({ motionVersion: 2 });
        r.steps.push(['synth2-before-estimate', engine.synth(t, undefined)]);
        r.steps.push(['est2', engine.estimate()]);
        r.steps.push(['synth2', engine.synth(t, undefined)]);
        const q3 = px();
        r.diff = { pKeptVsOwn: maxd(p1, p2), qLazyVsOwn: maxd(q1, q2), qRoundTrip: maxd(q1, q3), qVsP: maxd(q1, p1) };
        r.glError = gl.getError();
      } catch (e) { r.error = err(e); }
      return r;
    },
  };
})();
