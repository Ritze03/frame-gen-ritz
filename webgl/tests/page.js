// Runs inside the browser page, after the UNCHANGED framegen.js has attached globalThis.__fg.
// Exposes window.__golden. All data crosses the Node<->page boundary as base64 strings.
(() => {
  const b64ToBytes = (s) => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
  const bytesToB64 = (u8) => {
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  };
  const pack = (ta) => ({
    type: ta instanceof Float32Array ? 'f4' : ta instanceof Uint32Array ? 'u4' : ta instanceof Uint16Array ? 'u2' : 'u1',
    b64: bytesToB64(new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength)),
  });
  const err = (e) => ({ error: String((e && e.message) || e), stack: e && e.stack ? String(e.stack).split('\n').slice(0, 6).join('\n') : '' });

  let gl = null, engine = null, canvas = null;

  function read(name) {
    try {
      const r = engine.debugRead(name);
      return { name: r.name, width: r.width, height: r.height, channels: r.channels, ...pack(r.data) };
    } catch (e) { return err(e); }
  }

  window.__golden = {
    init({ sources, settings }) {
      canvas = (typeof OffscreenCanvas !== 'undefined') ? new OffscreenCanvas(64, 64) : Object.assign(document.createElement('canvas'), { width: 64, height: 64 });
      gl = canvas.getContext('webgl2', { antialias: false, alpha: true, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
      if (!gl) return { fatal: 'getContext("webgl2") returned null' };
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      const info = {
        renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
        version: gl.getParameter(gl.VERSION),
        extensions: Object.fromEntries(['EXT_color_buffer_float', 'EXT_float_blend', 'OES_texture_float_linear'].map((n) => [n, !!gl.getExtension(n)])),
        maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS),
      };
      if (typeof globalThis.__fg !== 'object' || !globalThis.__fg) return { fatal: 'globalThis.__fg is not defined after loading framegen.js', info };
      info.fgVersion = __fg.version;
      info.shaderFiles = Array.from(__fg.shaderFiles || []);
      try { info.support = __fg.checkSupport(gl); } catch (e) { return { fatal: 'checkSupport threw: ' + err(e).error, info }; }
      try { engine = __fg.createEngine(gl, sources, settings); } catch (e) { return { fatal: 'createEngine threw: ' + err(e).error, info }; }
      return { info };
    },

    setSettings(partial) { try { return engine.setSettings(partial); } catch (e) { return err(e); } },

    // motionVersion switch (library rule: a switch drops the estimate): V2 out, switch to V1 -> synth must be false until the next
    // estimate, V1 out, back to V2 -> false again, then the same V2 out as the first time. Returns the outputs for comparison.
    switchTest({ w, h, a, c, t }) {
      const r = { steps: [] };
      try {
        const px = () => { const o = engine.debugRead('out'); return Array.from(o.data); };
        engine.setSettings({ motionVersion: 2 });
        engine.pushFrame({ width: w, height: h, data: b64ToBytes(a) });
        engine.pushFrame({ width: w, height: h, data: b64ToBytes(c) });
        r.steps.push(['est2', engine.estimate()]);
        r.steps.push(['synth2', engine.synth(t, undefined)]);
        const o2 = px();
        engine.setSettings({ motionVersion: 1 });
        r.steps.push(['synth1-before-estimate', engine.synth(t, undefined)]);
        r.steps.push(['est1', engine.estimate()]);
        r.steps.push(['synth1', engine.synth(t, undefined)]);
        const o1 = px();
        engine.setSettings({ motionVersion: 2 });
        r.steps.push(['synth2-before-estimate', engine.synth(t, undefined)]);
        r.steps.push(['est2b', engine.estimate()]);
        r.steps.push(['synth2b', engine.synth(t, undefined)]);
        const o2b = px();
        let d21 = 0, d22 = 0;
        for (let i = 0; i < o2.length; i++) { d22 = Math.max(d22, Math.abs(o2[i] - o2b[i])); d21 = Math.max(d21, Math.abs(o2[i] - o1[i])); }
        r.maxDiffV2vsV2 = d22; r.maxDiffV2vsV1 = d21;
        r.glError = gl.getError();
      } catch (e) { r.error = err(e); }
      return r;
    },

    levelLayout(w, h, s) { try { return __fg.levelLayout(w, h, s); } catch (e) { return err(e); } },

    // One test pair: push A, C, estimate, then synth per t into an own texture-backed FBO.
    runPair({ w, h, a, c, ts, pairNames, tNames }) {
      const out = { calls: {}, est: {}, t: {} };
      const call = (k, f) => { try { out.calls[k] = f(); return true; } catch (e) { out.calls[k] = err(e); return false; } };
      if (!call('pushFrame(A)', () => engine.pushFrame({ width: w, height: h, data: b64ToBytes(a) }))) return out;
      if (!call('pushFrame(C)', () => engine.pushFrame({ width: w, height: h, data: b64ToBytes(c) }))) return out;
      out.info = engine.info ? JSON.parse(JSON.stringify(engine.info)) : null;
      if (!call('estimate', () => engine.estimate())) return out;
      for (const n of pairNames) out.est[n] = read(n);
      // own FBO (texture-backed, top-first, LAYOUT 5.1): never read the canvas
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      for (const t of ts) {
        const r = { calls: {}, stages: {} };
        out.t[t] = r;
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.clearColor(1, 0, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT); // magenta: makes a skipped blit obvious
        try { r.calls.synth = engine.synth(t, fbo, { x: 0, y: 0, width: w, height: h }); } catch (e) { r.calls.synth = err(e); continue; }
        try { r.stats = JSON.parse(JSON.stringify(engine.stats())); } catch (e) { r.stats = err(e); }
        for (const n of tNames) r.stages[n] = read(n);
        const px = new Uint8Array(w * h * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
        r.fbo = { width: w, height: h, channels: 4, ...pack(px) };
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      try { out.passCounts = engine.passCounts ? JSON.parse(JSON.stringify(engine.passCounts())) : null; } catch (e) { out.passCounts = err(e); }
      const glErr = gl.getError();
      out.glError = glErr;
      gl.deleteFramebuffer(fbo); gl.deleteTexture(tex);
      return out;
    },
  };
})();
