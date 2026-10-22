// Runs inside the test page (http://fg.test/, served by xbackend.mjs / bench.mjs through Playwright routes) after framegen.js attached
// globalThis.__fg. Exposes window.__x: init, run (one pair x settings: push, estimate, synth per t, compare with the Vulkan reference
// in the page, return small metrics), switchTest, bench. Files are fetched from /w/<path relative to $FG_WORK>.
(() => {
  let gl = null, engine = null;
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
      const canvas = new OffscreenCanvas(64, 64);
      gl = canvas.getContext('webgl2', { antialias: false, alpha: true, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: 'high-performance' });
      if (!gl) return { fatal: 'getContext("webgl2") returned null' };
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      const info = { renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER), maxTex: gl.getParameter(gl.MAX_TEXTURE_SIZE) };
      try { info.support = __fg.checkSupport(gl); engine = __fg.createEngine(gl, sources, settings); } catch (e) { return { fatal: err(e).error, info }; }
      return { info };
    },
    setSettings(p) { try { return engine.setSettings(p); } catch (e) { return err(e); } },

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
              if (maps.fscale >= 2) {
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
