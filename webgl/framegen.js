/*
 * framegen.js - WebGL2 engine host (I6). Implements extension/engine/LAYOUT.md sections 2, 3 and 6.
 *
 * Classic script (no modules, no build, no fetch). Attaches to globalThis.__fg (create-or-extend).
 * Works as a Firefox MAIN-world content script and in a plain harness page.
 *
 * GL state contract (LAYOUT 6, "GL state"):
 *   - Every public call that draws (pushFrame, estimate, synth, presentFrame) takes over the context:
 *     DITHER/BLEND/DEPTH_TEST/STENCIL_TEST/SCISSOR_TEST/CULL_FACE are disabled, colorMask is all true,
 *     the engine's own empty VAO / program / FBO / textures are bound, texture units 0..6 and
 *     ACTIVE_TEXTURE are clobbered. Nothing is restored.
 *   - After synth(t, fb, vp) / presentFrame(which, fb, vp) with a framebuffer given: `fb` (null = canvas)
 *     is bound as FRAMEBUFFER, gl.viewport = vp, the blit program is current, engine VAO bound.
 *   - After synth(t, undefined): the engine's `out` FBO is bound with viewport (0,0,W,H).
 *   - estimate() with settings.motion 'learned' also leaves a learned net's weight buffer bound to UNIFORM_BUFFER (the generic
 *     binding and indexed binding points 0 and 1).
 *   - pushFrame leaves UNPACK_FLIP_Y_WEBGL=false, UNPACK_PREMULTIPLY_ALPHA_WEBGL=false and, for raw
 *     {data} sources, UNPACK_ALIGNMENT=1.
 *   - debugRead/stats bind FBOs and change PACK state only transiently; they are not hot-path calls.
 * The hot path performs no readPixels / getError unless settings.debug is true.
 */
(function (root) {
  'use strict';

  const fg = root.__fg || (root.__fg = {});

  // ---------------------------------------------------------------------------------------------
  // Static tables
  // ---------------------------------------------------------------------------------------------

  fg.version = '3';   // 3: the learned motion estimator (settings motion / motionNet, __fg.nets, checkSupport learned); 2: V2 Quality,
  // RGBA16F input, synthBlur, frameSizeError, checkSupport sizeError / quality (API of the earlier versions unchanged otherwise)
  fg.requiredExtensions = ['EXT_color_buffer_float', 'EXT_float_blend', 'OES_texture_float_linear'];

  // Program table (LAYOUT 3.2). Sampler order = texture unit order. `vert` defaults to 'vert.glsl'.
  const PROGRAMS = [
    ['luma', 'estimate/luma.frag.glsl', null, ['uFrame']],
    ['down', 'estimate/down.frag.glsl', null, ['uSrc']],
    ['medhist', 'estimate/medhist.frag.glsl', null, ['uParent']],
    ['medsel', 'estimate/medsel.frag.glsl', null, ['uHist']],
    ['seed_sad', 'estimate/seed_sad.frag.glsl', null, ['uLumaA', 'uLumaC', 'uParent', 'uMedian']],
    ['seed_pick', 'estimate/seed_pick.frag.glsl', null, ['uSeedSad', 'uParent', 'uMedian']],
    ['search_sad', 'estimate/search_sad.frag.glsl', null, ['uLumaA', 'uLumaC', 'uSeed']],
    ['search_pick', 'estimate/search_pick.frag.glsl', null, ['uSearchSad', 'uSeed']],
    ['median3', 'estimate/median3.frag.glsl', null, ['uRaw']],
    ['subpix_sad', 'estimate/subpix_sad.frag.glsl', null, ['uLumaA', 'uLumaC', 'uBase']],
    ['subpix_pick', 'estimate/subpix_pick.frag.glsl', null, ['uSubSad', 'uBase']],
    ['reduce_sad', 'estimate/reduce_sad.frag.glsl', null, ['uSrc']],
    ['pf_err', 'estimate/pf_err.frag.glsl', null, ['uLumaA', 'uLumaC', 'uBlockField']],
    ['pf_select', 'estimate/pf_select.frag.glsl', null, ['uPfErr0', 'uPfErr1', 'uBlockField']],
    ['lookup', 'synth/lookup.frag.glsl', null, ['uField']],
    ['error', 'synth/error.frag.glsl', null, ['uPrev', 'uCurr', 'uField']],
    ['hbox', 'synth/hbox.frag.glsl', null, ['uE']],
    ['vbox', 'synth/vbox.frag.glsl', null, ['uH']],
    ['reduce_untrusted', 'synth/reduce_untrusted.frag.glsl', null, ['uSrc']],
    ['final', 'synth/final.frag.glsl', null,
      ['uPrev', 'uCurr', 'uWarpA', 'uWarpC', 'uTrust', 'uSadSum', 'uUntrusted']],
    ['blit', 'synth/blit.frag.glsl', null, ['uSrc']],
  ];

  // Motion version 2 programs, compiled when V2 is first selected (createEngine / setSettings). Optional 5th element: a shared
  // include assembled between common.glsl and the program file; optional 6th: #defines put right after common.glsl (see assemble()).
  const V2_INC = 'v2/morph_common.inc.glsl';
  const Q_INC = 'v2/q_common.inc.glsl';
  const Q_DEF = '#define V2_QUALITY 1\n';
  const MORPH_S = ['uPrev', 'uCurr', 'uVf', 'uSmask', 'uEC'];
  const Q_S = ['uQf', 'uCov'];
  const V2_PROGRAMS = [
    ['v2_static_raw', 'v2/static_raw.frag.glsl', null, ['uLumaA', 'uLumaC']],
    ['v2_static_s', 'v2/static_s.frag.glsl', null, ['uRaw']],
    ['v2_static_cell', 'v2/static_cell.frag.glsl', null, ['uSmask']],
    ['v2_search_pick0', 'v2/search_pick0.frag.glsl', null, ['uSearchSad', 'uSeed']],
    ['v2_repick_sad', 'v2/repick_sad.frag.glsl', null, ['uLumaA', 'uLumaC', 'uVfit']],
    ['v2_repick_pick', 'v2/repick_pick.frag.glsl', null, ['uRepSad', 'uVfit']],
    ['v2_refine_sad', 'v2/refine_sad.frag.glsl', null, ['uLumaA', 'uLumaC', 'uBase']],
    ['v2_refine_pick', 'v2/refine_pick.frag.glsl', null, ['uRepSad', 'uBase']],
    ['v2_clean', 'v2/clean.frag.glsl', null, ['uField']],
    ['v2_enclose', 'v2/enclose.frag.glsl', null, ['uS8', 'uVf']],
    ['v2_em', 'v2/em.frag.glsl', null, ['uPrev', 'uCurr', 'uVf', 'uSmask', 'uEC'], V2_INC],
    ['v2_group', 'v2/group.frag.glsl', null, ['uPrev', 'uCurr', 'uVf', 'uSmask', 'uEC', 'uEm'], V2_INC],
    ['v2_morph', 'v2/morph.frag.glsl', null, ['uPrev', 'uCurr', 'uVf', 'uSmask', 'uEC', 'uEm', 'uGroup'], V2_INC],
    ['v2_tile', 'v2/tile.frag.glsl', null, ['uE']],
    ['v2_final', 'v2/final.frag.glsl', null, ['uPrev', 'uCurr', 'uMorph', 'uSadSum', 'uUntrusted']],
    ['blur_accum', 'v2/blur_accum.frag.glsl', null, ['uSrc']],
    ['blur_resolve', 'v2/blur_resolve.frag.glsl', null, ['uAcc']],
    // V2 Quality (the coverage morph): its estimate tail and its synthesis (the Performance programs with V2_QUALITY)
    ['v2_qfield', 'v2/qfield.frag.glsl', null, ['uLumaA', 'uLumaC', 'uVf'], Q_INC],
    ['v2_qclaim', 'v2/qclaim.frag.glsl', null, ['uLumaA', 'uLumaC', 'uQf'], Q_INC],
    ['v2_qown_z', 'v2/qown_z.frag.glsl', null, ['uQf']],
    ['v2_qown', 'v2/qown.frag.glsl', 'v2/qown.vert.glsl', ['uClaim']],
    ['v2_qvis', 'v2/qvis.frag.glsl', null, ['uClaim', 'uOwnA', 'uOwnC'], Q_INC],
    ['v2_qcov', 'v2/qcov.frag.glsl', null, ['uVis', 'uQf'], Q_INC],
    ['v2_qem', 'v2/em.frag.glsl', null, MORPH_S.concat(Q_S), V2_INC, Q_DEF],
    ['v2_qgroup', 'v2/group.frag.glsl', null, MORPH_S.concat(Q_S, ['uEm']), V2_INC, Q_DEF],
    ['v2_qmorph', 'v2/morph.frag.glsl', null, MORPH_S.concat(Q_S, ['uEm', 'uGroup']), V2_INC, Q_DEF],
  ];

  // The learned motion estimator (settings.motion 'learned', V1 only; ml-fg export/SPEC.md), compiled the first time it is selected.
  // The convs are one program per shape. motionPrecision 'fp32' (conv.frag.glsl: ML_CIN RGBA32F input textures, ML_COUT output
  // textures, ML_LAST the 2-channel output layer): a net of width w (G = w / 4 textures per activation) uses ml_conv4x3 (layer 0: the
  // 14-channel input in 4 textures), ml_conv<G>x3 (layers 1-3) and ml_last<G>; G = 6 (e2v3) splits each layer into two passes of 3
  // output textures. 'fp16' (conv_h.frag.glsl, 8 half floats per RGBA32UI texel): ml_h<CIN>g<CG>o<COUT>[L], one pass per layer.
  const ML_INC = 'learned/ml_common.inc.glsl';
  function mlConv(cin, cout, last) {
    const s = [];
    for (let j = 0; j < cin; j++) s.push('uIn' + j);
    if (last) s.push('uHc');
    s.push('uDup');
    return [last ? 'ml_last' + cin : 'ml_conv' + cin + 'x' + cout, 'learned/conv.frag.glsl', null, s, ML_INC,
      '#define ML_CIN ' + cin + '\n#define ML_COUT ' + cout + '\n#define ML_LAST ' + (last ? 1 : 0) + '\n'];
  }
  // cin packed input textures, cg 4-channel input groups, cout 4-channel output groups
  function mlConvH(cin, cg, cout, last) {
    const s = [];
    for (let j = 0; j < cin; j++) s.push('uIn' + j);
    if (last) s.push('uHc');
    s.push('uDup');
    return ['ml_h' + cin + 'g' + cg + 'o' + cout + (last ? 'L' : ''), 'learned/conv_h.frag.glsl', null, s, ML_INC,
      '#define ML_CIN ' + cin + '\n#define ML_CG ' + cg + '\n#define ML_COUT ' + cout + '\n#define ML_COUTA ' + Math.min(cout, 3) +
      '\n#define ML_LAST ' + (last ? 1 : 0) + '\n'];
  }
  // the program of layer k (of depth + 1) of a net with G groups per activation, at precision prec
  function mlLayerProg(prec, G, k, depth) {
    const last = k === depth;
    if (prec === 'fp32') return last ? 'ml_last' + G : 'ml_conv' + (k === 0 ? 4 : G) + 'x3';
    const cin = k === 0 ? 2 : Math.ceil(G / 2);
    return 'ml_h' + cin + 'g' + (k === 0 ? 4 : G) + 'o' + (last ? 1 : G) + (last ? 'L' : '');
  }
  const ML_PROGRAMS = [
    ['ml_lv0', 'learned/lv0.frag.glsl', null, ['uPrev', 'uCurr'], ML_INC],
    ['ml_down', 'learned/down.frag.glsl', null, ['uSrc']],
    ['ml_match', 'learned/match.frag.glsl', null, ['uPyr', 'uHc', 'uDup'], ML_INC],
    ['ml_match_h', 'learned/match.frag.glsl', null, ['uPyr', 'uHc', 'uDup'], ML_INC, '#define ML_PACKED 1\n'],
    ['ml_sad', 'learned/sad.frag.glsl', null, ['uLv0', 'uField'], ML_INC],
    ['ml_sadout', 'learned/sadout.frag.glsl', null, ['uSrc', 'uDup'], ML_INC],
    mlConv(4, 3, false), mlConv(3, 3, false), mlConv(6, 3, false), mlConv(3, 1, true), mlConv(6, 1, true),
    mlConvH(2, 4, 3, false), mlConvH(2, 4, 6, false), mlConvH(2, 3, 3, false), mlConvH(3, 6, 6, false), mlConvH(2, 3, 1, true),
    mlConvH(3, 6, 1, true),
  ];
  const ML_DUP_DIFF = 0.6;                   // learned/ml_common.inc.glsl's (the duplicate-pair test)
  const ML_NETS = ['e2v3', 'e1v3'];          // settings.motionNet values (the first is the default); weights: webgl/nets/<name>.js
  const ML_BLOCK_MAX = (3 * 9 * 6 * 16 + 3 * 4) * 4;   // bytes of the largest weight block (conv6x3, conv_h's MlW / MlW2)

  fg.shaderFiles = (function () {
    const files = ['common.glsl', 'vert.glsl'];
    for (const [, frag, vert, , inc] of PROGRAMS.concat(V2_PROGRAMS, ML_PROGRAMS)) {
      for (const f of [vert, inc, frag]) if (f && files.indexOf(f) < 0) files.push(f);
    }
    return files;
  })();

  // The format #defines of the fragment shaders (common.glsl "frame formats"): none for 'rgba8'.
  function formatDefines(s) {
    if (s.inputFormat !== 'rgba16f') return '';
    return '#define FG_F16 1\n#define FG_XF_LINEAR ' + (s.transfer === 'linear' ? 1 : 0) + '\n#define FG_XF_WHITE ' +
      Number(s.linearWhiteNits).toFixed(6) + '\n';
  }

  // The exact GLSL the host compiles for each program: { name, vertexFile, vertex, fragmentFile, fragment }. A fragment shader is
  // common.glsl (with the format defines after its #version line) + the program's defines + "#line 1" + include + "#line 1" + file;
  // vertex shaders are standalone. Used by createEngine and by webgl/tests/validate.mjs (glslangValidator): both compile the same text.
  function assemble(list, srcOf, fmtDefs) {
    let common = srcOf('common.glsl');
    if (fmtDefs) { const nl = common.indexOf('\n') + 1; common = common.slice(0, nl) + fmtDefs + '#line 2\n' + common.slice(nl); }
    return list.map(function (e) {
      const name = e[0], frag = e[1], vert = e[2] || 'vert.glsl', inc = e[4], defs = e[5] || '';
      return {
        name: name, vertexFile: vert, vertex: srcOf(vert), fragmentFile: frag,
        fragment: common + '\n' + defs + '#line 1\n' + (inc ? srcOf(inc) + '\n#line 1\n' : '') + srcOf(frag),
      };
    });
  }
  // settings: optional (partial) settings whose format keys select the variant (default rgba8)
  fg.assemblePrograms = function (sources, settings) {
    return assemble(PROGRAMS.concat(V2_PROGRAMS, ML_PROGRAMS), function (k) {
      if (typeof sources[k] !== 'string') throw new Error('framegen: missing shader source "' + k + '"');
      return sources[k];
    }, formatDefines(mergeSettings(DEFAULTS, settings)));
  };

  // Uniform names the host may set. Locations are looked up once at link time.
  const U_STATIC = ['uFrameSize', 'uFlowScale', 'uNumBlocks', 'uLambda', 'uSmoothBonus', 'uZeroBonus',
    'uCutSad', 'uTrust0', 'uTrustScale', 'uGlobalFrac', 'uTrustMid'];
  const U_TIME = ['uT', 'uS', 'uWA', 'uWC', 'uNearC', 'uMid'];
  const U_PASS = ['uBlocks', 'uCount', 'uRadius', 'uStep', 'uSrcSize', 'uChan', 'uMode',
    'uVpOrigin', 'uVpSize', 'uFlipY', 'uExact', 'uStaticOff', 'uSide', 'uFan', 'uWeight', 'uWsum', 'uSize', 'uTop'];
  const U_ALL = U_STATIC.concat(U_TIME, U_PASS);

  const DEFAULTS = {
    flowScale: 2,
    searchPenalty: 0.5,
    smoothBonus: 0.5,
    hudBonus: 1.0,
    sceneCutSad: 30.0,
    trustLow: 12.0,
    trustHigh: 40.0,
    globalFallback: 0.15,
    motionVersion: 1,       // 1 = V1 (block matching + per-pixel trust, proto/interp.py), 2 = V2 (the morphing interpolator)
    v2Mode: 'quality',      // V2 only: 'quality' (the coverage morph, library default) or 'performance' (the grid morph)
    v2StaticMask: true,     // V2 only: keep static HUD / text / subtitles in place (Settings::v2StaticMask)
    blurSamples: 0,         // synthBlur (V2): sub-frames per output, 0 / 1 = off, 2..16 (Settings::blurSamples)
    blurWeights: 'gaussian', // synthBlur: 'gaussian' or 'even' (Settings::blurWeights)
    motion: 'search',       // V1 only: 'search' (V1's block search, proto/interp.py) or 'learned' (the learned estimator: needs flowScale 2
                            // and the net's weights, webgl/nets/<motionNet>.js; ignored by V2)
    motionNet: 'e2v3',      // motion 'learned': 'e2v3' (width 24, 95,530 weights) or 'e1v3' (width 12, 28,330 weights, ~3.4x cheaper)
    motionPrecision: 'fp16', // motion 'learned': storage of the net's activations, 'fp16' (8 halves per RGBA32UI texel) or 'fp32' (RGBA32F, exact);
                            // the field, the pyramid, the weights and all arithmetic are float32 either way
    // ---- structural (re-allocate textures and recompile the programs; drop the frame history) ----
    inputFormat: 'rgba8',   // frame format: 'rgba8' (RGBA8 frames and output) or 'rgba16f' (HDR / float: RGBA16F frames and output)
    transfer: 'encoded',    // rgba16f only: 'encoded' (display-encoded values: PQ, HLG, sRGB, ...) or 'linear' (scRGB / linear light)
    linearWhiteNits: 80,    // rgba16f + linear: the luminance of the value 1.0 (80 = scRGB), (0, 10000]
    debug: false,
  };
  const MH_ROWS = 4;   // parent rows per band of the median histogram (estimate/medhist.frag.glsl's)
  const STRING_KEYS = { v2Mode: ['quality', 'performance'], blurWeights: ['gaussian', 'even'], inputFormat: ['rgba8', 'rgba16f'], transfer: ['encoded', 'linear'],
    motion: ['search', 'learned'], motionNet: ML_NETS, motionPrecision: ['fp16', 'fp32'] };
  const BOOL_KEYS = { debug: 1, v2StaticMask: 1 };
  fg.defaultSettings = function () { return Object.assign({}, DEFAULTS); };

  // round-half-to-even (= cv2.saturate_cast<int> = FG:346 rintHalfEven)
  function rhe(x) {
    const f = Math.floor(x), d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return (f % 2 === 0) ? f : f + 1;
  }

  // LAYOUT 2: pyramid sizes.
  fg.levelLayout = function (W, H, flowScale) {
    if (!(W >= 1 && H >= 1 && Math.floor(W) === W && Math.floor(H) === H)) {
      throw new RangeError('framegen: levelLayout: bad frame size ' + W + 'x' + H);
    }
    if (flowScale !== 1 && flowScale !== 2 && flowScale !== 4) {
      throw new RangeError('framegen: levelLayout: flowScale must be 1, 2 or 4 (got ' + flowScale + ')');
    }
    const fw = rhe(W / flowScale), fh = rhe(H / flowScale);
    if (fw < 1 || fh < 1) throw new RangeError('framegen: levelLayout: frame too small for flowScale');
    const levels = [];
    let w = fw, h = fh;
    for (;;) {
      levels.push({ w: w, h: h, bw: Math.ceil(w / 8), bh: Math.ceil(h / 8), radius: 2 });
      if (Math.floor(Math.min(w, h) / 2) >= 48) { w = rhe(w / 2); h = rhe(h / 2); } else break;
    }
    levels[levels.length - 1].radius = 6;
    return { fw: fw, fh: fh, levels: levels };
  };

  // Why a frame of w x h cannot run with these settings on this context (a string), or null: the library's minimum size 16 x 16,
  // and every texture the engine allocates (frames, the per-level search tiles) within MAX_TEXTURE_SIZE.
  fg.minSize = 16;
  fg.frameSizeError = function (gl, w, h, settings) {
    if (!(Number.isInteger(w) && Number.isInteger(h))) return 'frame size ' + w + 'x' + h + ' is not integer';
    if (w < fg.minSize || h < fg.minSize) return 'frame ' + w + 'x' + h + ' is below the minimum ' + fg.minSize + 'x' + fg.minSize;
    const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const S = (settings && settings.flowScale) || DEFAULTS.flowScale;
    let need = Math.max(w, h);
    for (const lv of fg.levelLayout(w, h, S).levels) need = Math.max(need, (2 * lv.radius + 1) * Math.max(lv.bw, lv.bh));
    if (need > max) return 'frame ' + w + 'x' + h + ' at flowScale ' + S + ' needs ' + need + ' px textures, MAX_TEXTURE_SIZE is ' + max;
    return null;
  };

  // V2 Quality's coverage splat (v2/qown.vert.glsl) needs MIN blending on an R32F target to keep exactly the smallest of several
  // positive floats (their bit patterns). Not every driver blends 32-bit floats with MIN exactly, so it is tried: three uint words
  // (as float bits, the splat's range >= 2^23) are drawn as points of size 1 into one R32F texel in the order big, small, mid; the
  // texel must read back as the small word bit for bit. -> null (works) or the reason. Cached per context (not while it is lost).
  // Saves and restores every binding / state it touches; drains pending GL errors first (they would be read as its own).
  const qProbeCache = typeof WeakMap === 'function' ? new WeakMap() : null;
  function probeQuality(gl) {
    if (qProbeCache && qProbeCache.has(gl)) return qProbeCache.get(gl);
    if (gl.isContextLost && gl.isContextLost()) return 'context lost';
    const VS = '#version 300 es\nprecision highp float;\nprecision highp int;\nuniform uint uW[3];\nflat out uint vW;\n' +
      'void main() { vW = uW[gl_VertexID]; gl_PointSize = 1.0; gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n';
    const FS = '#version 300 es\nprecision highp float;\nprecision highp int;\nflat in uint vW;\nlayout(location = 0) out vec4 o;\n' +
      'void main() { o = vec4(uintBitsToFloat(vW), 0.0, 0.0, 0.0); }\n';
    const WORDS = [0x0FA12345, 0x00923457, 0x05834567], WANT = 0x00923457;
    while (gl.getError() !== gl.NO_ERROR) { /* drain foreign errors */ }
    const P = gl.getParameter.bind(gl);
    const saved = {
      draw: P(gl.DRAW_FRAMEBUFFER_BINDING), read: P(gl.READ_FRAMEBUFFER_BINDING), prog: P(gl.CURRENT_PROGRAM),
      vao: P(gl.VERTEX_ARRAY_BINDING), vp: P(gl.VIEWPORT), blend: gl.isEnabled(gl.BLEND), scissor: gl.isEnabled(gl.SCISSOR_TEST),
      discard: gl.isEnabled(gl.RASTERIZER_DISCARD), eqRgb: P(gl.BLEND_EQUATION_RGB), eqA: P(gl.BLEND_EQUATION_ALPHA),
      mask: P(gl.COLOR_WRITEMASK), clear: P(gl.COLOR_CLEAR_VALUE), active: P(gl.ACTIVE_TEXTURE), pack: P(gl.PIXEL_PACK_BUFFER_BINDING),
    };
    gl.activeTexture(gl.TEXTURE0);
    const tex0 = P(gl.TEXTURE_BINDING_2D);
    let tex = null, fbo = null, vs = null, fs = null, prog = null, vao = null, why = null;
    try {
      tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, 1, 1);
      fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (st !== gl.FRAMEBUFFER_COMPLETE) return (why = 'R32F not color-renderable (framebuffer status 0x' + st.toString(16) + ')');
      const sh = (type, src) => { const o = gl.createShader(type); gl.shaderSource(o, src); gl.compileShader(o); return o; };
      vs = sh(gl.VERTEX_SHADER, VS); fs = sh(gl.FRAGMENT_SHADER, FS);
      prog = gl.createProgram();
      gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return (why = 'MIN blend probe program failed to link: ' + String(gl.getProgramInfoLog(prog) || '').trim());
      vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.useProgram(prog);
      gl.uniform1uiv(gl.getUniformLocation(prog, 'uW[0]'), new Uint32Array(WORDS));
      gl.viewport(0, 0, 1, 1);
      gl.disable(gl.SCISSOR_TEST); gl.disable(gl.RASTERIZER_DISCARD);
      gl.colorMask(true, true, true, true);
      gl.clearColor(3e38, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.MIN);
      gl.drawArrays(gl.POINTS, 0, 3);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      const px = new Float32Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
      const e = gl.getError();
      if (e !== gl.NO_ERROR) return (why = 'MIN blend on R32F: GL error 0x' + e.toString(16));
      const got = new Uint32Array(px.buffer)[0];
      if (got !== WANT) return (why = 'MIN blend on R32F: kept 0x' + got.toString(16) + ', want 0x' + WANT.toString(16));
      return null;
    } catch (ex) {
      return (why = 'MIN blend on R32F: ' + String((ex && ex.message) || ex));
    } finally {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, saved.draw); gl.bindFramebuffer(gl.READ_FRAMEBUFFER, saved.read);
      gl.useProgram(saved.prog); gl.bindVertexArray(saved.vao);
      gl.viewport(saved.vp[0], saved.vp[1], saved.vp[2], saved.vp[3]);
      (saved.blend ? gl.enable : gl.disable).call(gl, gl.BLEND);
      (saved.scissor ? gl.enable : gl.disable).call(gl, gl.SCISSOR_TEST);
      (saved.discard ? gl.enable : gl.disable).call(gl, gl.RASTERIZER_DISCARD);
      gl.blendEquationSeparate(saved.eqRgb, saved.eqA);
      gl.colorMask(saved.mask[0], saved.mask[1], saved.mask[2], saved.mask[3]);
      gl.clearColor(saved.clear[0], saved.clear[1], saved.clear[2], saved.clear[3]);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, saved.pack);
      gl.bindTexture(gl.TEXTURE_2D, tex0); gl.activeTexture(saved.active);
      gl.deleteVertexArray(vao); gl.deleteProgram(prog); gl.deleteShader(vs); gl.deleteShader(fs);
      gl.deleteFramebuffer(fbo); gl.deleteTexture(tex);
      while (gl.getError() !== gl.NO_ERROR) { /* the probe's own */ }
      if (qProbeCache && !(gl.isContextLost && gl.isContextLost())) qProbeCache.set(gl, why);
    }
  }

  // -> { ok, missing, sizeError, quality, qualityError, learned, learnedError }. missing: the GL features this context lacks (names
  // only; 'webgl2' alone if it is no WebGL2 context). sizeError: the optional frame = { width, height, flowScale } checked by
  // fg.frameSizeError, else null. ok = nothing missing and no sizeError. quality: V2 Quality can run (its MIN-blend splat works,
  // probeQuality); when false, qualityError says why. V1 and V2 Performance do not need it, so it does not affect ok. learned: the
  // learned motion estimator (settings.motion 'learned') can run: 4 draw buffers / colour attachments, uniform blocks of ML_BLOCK_MAX
  // bytes, 2 fragment uniform blocks, 8 fragment texture units (all WebGL2 core minimums, so false only on a non-conformant
  // context); does not affect ok either. (Whether a net's weights are loaded is fg.nets, not the context: createEngine / setSettings
  // say so.)
  fg.checkSupport = function (gl, frame) {
    const missing = [];
    let isGl2;
    if (!gl) isGl2 = false;
    else if (typeof root.WebGL2RenderingContext !== 'undefined') isGl2 = gl instanceof root.WebGL2RenderingContext;
    else isGl2 = typeof gl.texStorage2D === 'function' && typeof gl.createVertexArray === 'function';
    if (!isGl2) return { ok: false, missing: ['webgl2'], sizeError: null, quality: false, qualityError: 'no WebGL2 context', learned: false, learnedError: 'no WebGL2 context' };
    for (const name of fg.requiredExtensions) {
      if (!gl.getExtension(name)) missing.push(name);
    }
    if (!(gl.getParameter(gl.MAX_DRAW_BUFFERS) >= 3)) missing.push('MAX_DRAW_BUFFERS>=3');
    if (!(gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) >= 1)) missing.push('MAX_VERTEX_TEXTURE_IMAGE_UNITS>=1');
    const sizeError = frame ? fg.frameSizeError(gl, frame.width, frame.height, frame) : null;
    // (the probe needs the float render / blend extensions; without them nothing runs anyway)
    const qualityError = missing.length ? 'needs ' + missing.join(', ') : probeQuality(gl);
    const mlMissing = missing.slice();
    for (const [k, min, name] of [[gl.MAX_DRAW_BUFFERS, 4, 'MAX_DRAW_BUFFERS'], [gl.MAX_COLOR_ATTACHMENTS, 4, 'MAX_COLOR_ATTACHMENTS'],
      [gl.MAX_UNIFORM_BLOCK_SIZE, ML_BLOCK_MAX, 'MAX_UNIFORM_BLOCK_SIZE'], [gl.MAX_FRAGMENT_UNIFORM_BLOCKS, 2, 'MAX_FRAGMENT_UNIFORM_BLOCKS'],
      [gl.MAX_TEXTURE_IMAGE_UNITS, 8, 'MAX_TEXTURE_IMAGE_UNITS']]) {
      if (!(gl.getParameter(k) >= min)) mlMissing.push(name + '>=' + min);
    }
    const learnedError = mlMissing.length ? 'needs ' + mlMissing.join(', ') : null;
    return { ok: missing.length === 0 && !sizeError, missing: missing, sizeError: sizeError, quality: !qualityError, qualityError: qualityError,
      learned: !learnedError, learnedError: learnedError };
  };

  // ---------------------------------------------------------------------------------------------
  // Settings
  // ---------------------------------------------------------------------------------------------

  function mergeSettings(base, partial) {
    const s = Object.assign({}, base);
    if (partial) {
      for (const k of Object.keys(DEFAULTS)) {
        if (partial[k] !== undefined) s[k] = partial[k];
      }
    }
    if (s.flowScale !== 1 && s.flowScale !== 2 && s.flowScale !== 4) {
      throw new RangeError('framegen: settings.flowScale must be 1, 2 or 4 (got ' + s.flowScale + ')');
    }
    if (s.motionVersion !== 1 && s.motionVersion !== 2) {
      throw new RangeError('framegen: settings.motionVersion must be 1 or 2 (got ' + s.motionVersion + ')');
    }
    for (const k of Object.keys(DEFAULTS)) {
      if (BOOL_KEYS[k]) { s[k] = !!s[k]; continue; }
      if (STRING_KEYS[k]) {
        const v = String(s[k]).toLowerCase();
        if (STRING_KEYS[k].indexOf(v) < 0) throw new RangeError('framegen: settings.' + k + ' must be one of ' + STRING_KEYS[k].join(', ') + ' (got ' + s[k] + ')');
        s[k] = v;
        continue;
      }
      if (typeof s[k] !== 'number' || !isFinite(s[k])) throw new TypeError('framegen: settings.' + k + ' must be a finite number');
    }
    if (!(s.trustHigh >= s.trustLow + 1)) throw new RangeError('framegen: settings.trustHigh must be >= trustLow + 1');
    if (!(Number.isInteger(s.blurSamples) && s.blurSamples >= 0 && s.blurSamples <= 16)) throw new RangeError('framegen: settings.blurSamples must be an integer 0..16');
    if (!(s.linearWhiteNits > 0 && s.linearWhiteNits <= 10000)) throw new RangeError('framegen: settings.linearWhiteNits must be in (0, 10000]');
    if (s.motionVersion === 1 && s.motion === 'learned' && s.flowScale !== 2) {
      throw new RangeError("framegen: settings.motion 'learned' needs flowScale 2 (the net was trained at 2; got " + s.flowScale + ')');
    }
    return Object.freeze(s);
  }

  // the learned estimator is the effective motion estimate (V1 with motion 'learned'; V2 ignores the key)
  function mlOn(s) { return s.motionVersion === 1 && s.motion === 'learned'; }
  // which estimate the settings produce: a change drops the current estimate (like a motionVersion switch)
  function estKey(s) { return s.motionVersion === 2 ? 'v2' : mlOn(s) ? 'ml:' + s.motionNet + ':' + s.motionPrecision : 'v1'; }
  // the loaded net `name` (webgl/nets/<name>.js attaches fg.nets[name]) or null
  function mlNetDef(name) {
    const n = fg.nets && fg.nets[name];
    if (!n || typeof n.weights !== 'string' || n.levels !== 5 || n.inChannels !== 14 || n.padMultiple !== 16 || !(n.width === 12 || n.width === 24) ||
      !Array.isArray(n.stacks) || !Array.isArray(n.levelToStack)) return null;
    return n;
  }

  // ---------------------------------------------------------------------------------------------
  // Engine
  // ---------------------------------------------------------------------------------------------

  fg.createEngine = function (gl, sources, settingsIn) {
    const sup = fg.checkSupport(gl);
    if (!sup.ok) throw new Error('framegen: unsupported context, missing: ' + sup.missing.join(', '));
    if (!sources || typeof sources !== 'object') throw new TypeError('framegen: createEngine: sources object required');

    // V2 Quality is refused cleanly where its splat cannot work (checkSupport().quality); V1 and V2 Performance still run
    function qualityGate(s) {
      if (s.motionVersion === 2 && s.v2Mode === 'quality' && !sup.quality) {
        throw new RangeError("framegen: settings.v2Mode 'quality' is not supported on this context (" + sup.qualityError + "); use v2Mode 'performance' or motionVersion 1");
      }
      return s;
    }
    // the learned estimator: refused cleanly where the context lacks what it needs or its net's weights are not loaded
    function learnedGate(s) {
      if (!mlOn(s)) return s;
      if (!sup.learned) throw new RangeError("framegen: settings.motion 'learned' is not supported on this context (" + sup.learnedError + ')');
      if (!mlNetDef(s.motionNet)) {
        throw new RangeError("framegen: settings.motion 'learned': the weights of net '" + s.motionNet + "' are not loaded (load webgl/nets/" +
          s.motionNet + '.js next to framegen.js)');
      }
      return s;
    }
    let settings = learnedGate(qualityGate(mergeSettings(DEFAULTS, settingsIn)));
    let debug = settings.debug;
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);

    // ---- internal formats -------------------------------------------------------------------
    // read: how debugRead pulls the texture back (format, type, typed array constructor).
    const FMT = {
      RGBA8: { i: gl.RGBA8, f: gl.RGBA, t: gl.UNSIGNED_BYTE, ch: 4, rf: gl.RGBA, rt: gl.UNSIGNED_BYTE, A: Uint8Array },
      R8: { i: gl.R8, f: gl.RED, t: gl.UNSIGNED_BYTE, ch: 1, rf: gl.RGBA, rt: gl.UNSIGNED_BYTE, A: Uint8Array },
      R16UI: { i: gl.R16UI, f: gl.RED_INTEGER, t: gl.UNSIGNED_SHORT, ch: 1, rf: gl.RGBA_INTEGER, rt: gl.UNSIGNED_INT, A: Uint32Array },
      R32F: { i: gl.R32F, f: gl.RED, t: gl.FLOAT, ch: 1, rf: gl.RGBA, rt: gl.FLOAT, A: Float32Array },
      RG32F: { i: gl.RG32F, f: gl.RG, t: gl.FLOAT, ch: 2, rf: gl.RGBA, rt: gl.FLOAT, A: Float32Array },
      RGBA32UI: { i: gl.RGBA32UI, f: gl.RGBA_INTEGER, t: gl.UNSIGNED_INT, ch: 4, rf: gl.RGBA_INTEGER, rt: gl.UNSIGNED_INT, A: Uint32Array },
      RGBA32F: { i: gl.RGBA32F, f: gl.RGBA, t: gl.FLOAT, ch: 4, rf: gl.RGBA, rt: gl.FLOAT, A: Float32Array },
      RGBA32I: { i: gl.RGBA32I, f: gl.RGBA_INTEGER, t: gl.INT, ch: 4, rf: gl.RGBA_INTEGER, rt: gl.INT, A: Int32Array },
      RGBA16F: { i: gl.RGBA16F, f: gl.RGBA, t: gl.HALF_FLOAT, ch: 4, rf: gl.RGBA, rt: gl.FLOAT, A: Float32Array },
      R16F: { i: gl.R16F, f: gl.RED, t: gl.HALF_FLOAT, ch: 1, rf: gl.RGBA, rt: gl.FLOAT, A: Float32Array },
    };
    // the frame / output / warp texture format of settings.inputFormat
    function colorFmt() { return settings.inputFormat === 'rgba16f' ? 'RGBA16F' : 'RGBA8'; }

    // ---- programs ---------------------------------------------------------------------------
    function srcOf(key) {
      const s = sources[key];
      if (typeof s !== 'string') throw new Error('framegen: missing shader source "' + key + '"');
      return s;
    }

    function infoLog(log) { return String(log || '').replace(/\0+$/, '').trim(); }

    const P = {};            // name -> { name, prog, u, hasTime, samplers }
    const progList = [];
    let vao = null;
    let v2Built = false;
    let mlBuilt = false;
    function buildPrograms(list) {
      const shaders = [];    // { sh, file }
      const vertCache = {};
      const built = [];      // { name, frag, prog, samplers }
      function mkShader(type, file, src) {
        const sh = gl.createShader(type);
        shaders.push({ sh: sh, file: file });
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        return sh;
      }
      function cleanup() {
        for (const b of built) gl.deleteProgram(b.prog);
        for (const s of shaders) gl.deleteShader(s.sh);
      }
      try {
        const asm = assemble(list, srcOf, formatDefines(settings));
        for (let k = 0; k < list.length; k++) {
          const a = asm[k], samplers = list[k][3];
          const vsh = vertCache[a.vertexFile] || (vertCache[a.vertexFile] = mkShader(gl.VERTEX_SHADER, a.vertexFile, a.vertex));
          const fsh = mkShader(gl.FRAGMENT_SHADER, a.fragmentFile + (list[k][5] ? ' (' + a.name + ')' : ''), a.fragment);
          const prog = gl.createProgram();
          const name = a.name, frag = a.fragmentFile;
          built.push({ name: name, frag: frag, prog: prog, samplers: samplers });
          gl.attachShader(prog, vsh);
          gl.attachShader(prog, fsh);
          gl.linkProgram(prog);
        }
        // Status is read after everything was submitted so the driver can compile in parallel.
        for (const s of shaders) {
          if (!gl.getShaderParameter(s.sh, gl.COMPILE_STATUS)) {
            throw new Error('framegen: ' + s.file + ': ' + infoLog(gl.getShaderInfoLog(s.sh)));
          }
        }
        for (const b of built) {
          if (!gl.getProgramParameter(b.prog, gl.LINK_STATUS)) {
            throw new Error('framegen: ' + b.frag + ': link failed: ' + infoLog(gl.getProgramInfoLog(b.prog)));
          }
        }
      } catch (e) {
        cleanup();
        throw e;
      }
      for (const s of shaders) gl.deleteShader(s.sh);   // programs keep them alive until deleted
      for (const b of built) {
        const u = {};
        for (const n of U_ALL) {
          const loc = gl.getUniformLocation(b.prog, n);
          if (loc) u[n] = loc;
        }
        // the learned convs' weights (uniform blocks MlW, MlW2): binding points 0 and 1, a range of the net's buffer each per pass
        const wSize = [0, 0];
        ['MlW', 'MlW2'].forEach(function (bn, k) {
          const bi = gl.getUniformBlockIndex(b.prog, bn);
          if (bi === gl.INVALID_INDEX) return;
          gl.uniformBlockBinding(b.prog, bi, k);
          wSize[k] = gl.getActiveUniformBlockParameter(b.prog, bi, gl.UNIFORM_BLOCK_DATA_SIZE);
        });
        gl.useProgram(b.prog);
        b.samplers.forEach(function (sn, unit) {
          const loc = gl.getUniformLocation(b.prog, sn);
          if (loc) gl.uniform1i(loc, unit);   // (a sampler of a shared include that this program does not read is inactive: fine)
        });
        const p = {
          name: b.name, prog: b.prog, u: u, wSize: wSize,
          hasTime: U_TIME.some(function (n) { return !!u[n]; }),
        };
        P[b.name] = p;
        progList.push(p);
      }
      gl.useProgram(null);
    }
    buildPrograms(PROGRAMS);
    vao = gl.createVertexArray();
    // a change of the frame format (inputFormat / transfer / linearWhiteNits) recompiles every program built so far
    function rebuildPrograms() {
      for (const p of progList) gl.deleteProgram(p.prog);
      progList.length = 0;
      for (const k of Object.keys(P)) delete P[k];
      buildPrograms(PROGRAMS);
      if (v2Built) buildPrograms(V2_PROGRAMS);
      if (mlBuilt) buildPrograms(ML_PROGRAMS);
      staticDirty = true;
    }
    // V2 programs are compiled the first time V2 is selected, so a V1-only session never pays for them.
    function ensureV2Programs() {
      if (v2Built) return;
      buildPrograms(V2_PROGRAMS);
      v2Built = true;
      staticDirty = true;
    }
    // the learned estimator's programs likewise, the first time settings select it
    function ensureMlPrograms() {
      if (mlBuilt) return;
      buildPrograms(ML_PROGRAMS);
      mlBuilt = true;
      staticDirty = true;
    }

    // ---- state ------------------------------------------------------------------------------
    let disposed = false;
    let T = null;                  // texture set (see alloc)
    let info = null;
    let W = 0, H = 0, L = 0, FW = 0, FH = 0, levels = null;
    let ownedTex = [];             // { tex, fbo }
    let ownedFbo = [];
    let currS = 0, prevS = 1;      // slots; frame count since resize
    let count = 0;
    let estimatedPair = false;     // an estimate exists for the current (prev,curr)
    let haveEstimate = false;      // pair textures hold data (since resize)
    let synthDone = false, synthMid = false, lastT = null;
    let staticDirty = true;
    let qDone = false;             // V2 Quality's estimate tail has run for the current estimate
    let mlEst = false;             // the current estimate is the learned estimator's (stats: duplicate)
    const mlBufs = {};             // net name -> its weights in a uniform buffer (mlNetBuffer)
    if (settings.motionVersion === 2) ensureV2Programs();
    if (mlOn(settings)) ensureMlPrograms();

    // time-dependent uniform values of the current synth call
    let tT = 0, tS = 0, tWA = 0, tWC = 0, tNear = 0, tMid = 0;

    // pass counters
    let nDraw = 0, nClear = 0;
    const counts = { pushFrame: 0, estimate: 0, synth: 0, clears: 0 };

    // JS-side GL state cache; reset at the start of every public call.
    let curProg = null, curFbo, curUnit = -1;
    let vpX = -1, vpY = -1, vpW = -1, vpH = -1;
    const units = new Array(16).fill(null);

    function resetCache() {
      curProg = null; curFbo = undefined; curUnit = -1;
      vpX = vpY = vpW = vpH = -1;
      for (let i = 0; i < units.length; i++) units[i] = null;
    }

    function chk(name) {
      const e = gl.getError();
      if (e !== gl.NO_ERROR) throw new Error('framegen: GL error 0x' + e.toString(16) + ' in ' + name);
    }

    function alive() { if (disposed) throw new Error('framegen: engine disposed'); }

    function beginCall() {
      alive();
      resetCache();
      if (debug) { while (gl.getError() !== gl.NO_ERROR) { /* drain foreign errors */ } }
      gl.disable(gl.DITHER);
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.STENCIL_TEST);
      gl.disable(gl.SCISSOR_TEST);
      gl.disable(gl.CULL_FACE);
      gl.colorMask(true, true, true, true);
      gl.bindVertexArray(vao);
      if (staticDirty && info) applyStatics();
    }

    // ---- resource helpers -------------------------------------------------------------------
    function fboStatusCheck(label) {
      const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (st !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('framegen: framebuffer incomplete for ' + label + ' (status 0x' + st.toString(16) + ')');
      }
    }

    // Texture + single-attachment FBO (also used by debugRead). Completeness checked here, once.
    function mk(name, fmtKey, w, h, linear) {
      if (w > maxTex || h > maxTex) {
        throw new Error('framegen: texture ' + name + ' ' + w + 'x' + h + ' exceeds MAX_TEXTURE_SIZE ' + maxTex);
      }
      const fmt = FMT[fmtKey];
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, 1, fmt.i, w, h);
      const filt = linear ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
      const fbo = gl.createFramebuffer();
      const t = { name: name, tex: tex, fbo: fbo, w: w, h: h, fmt: fmt };
      ownedTex.push(t);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      fboStatusCheck('texture ' + name + ' (' + fmtKey + ' ' + w + 'x' + h + ')');
      return t;
    }

    function mkMrt(name, texes) {
      const fbo = gl.createFramebuffer();
      ownedFbo.push(fbo);
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      const bufs = [];
      for (let i = 0; i < texes.length; i++) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texes[i].tex, 0);
        bufs.push(gl.COLOR_ATTACHMENT0 + i);
      }
      gl.drawBuffers(bufs);
      fboStatusCheck('MRT ' + name + ' (' + texes.map(function (x) { return x.name; }).join(', ') + ')');
      return { name: name, fbo: fbo, w: texes[0].w, h: texes[0].h };
    }

    function freeTextures() {
      for (const t of ownedTex) { gl.deleteFramebuffer(t.fbo); gl.deleteTexture(t.tex); }
      for (const f of ownedFbo) gl.deleteFramebuffer(f);
      ownedTex = []; ownedFbo = [];
      T = null; info = null; levels = null;
      count = 0; currS = 0; prevS = 1;
      estimatedPair = false; haveEstimate = false; synthDone = false; synthMid = false; lastT = null; qDone = false; mlEst = false;
      resetCache();
    }

    // sizes of a sum-reduction chain down to 1x1: the first step's fan-in fan0 per axis, then rest (default 16; at least one step)
    function reduceChain(w, h, fan0, rest) {
      const sizes = [];
      let f = fan0 || 16;
      do {
        w = Math.ceil(w / f); h = Math.ceil(h / f);
        sizes.push([w, h, f]);
        f = rest || 16;
      } while (w > 1 || h > 1);
      return sizes;
    }
    function mkChain(name, w, h, fan0, rest) {
      return reduceChain(w, h, fan0, rest).map(function (sz, k) { const t = mk(name + '[' + k + ']', 'R32F', sz[0], sz[1]); t.fan = sz[2]; return t; });
    }
    // run a reduction chain: step 0 reads src (channel chan; reduce_untrusted: mode0 for the first step), the last output is the sum
    function runChain(prog, chain, src, first, rest) {
      for (let k = 0; k < chain.length; k++) {
        const s = k === 0 ? src : chain[k - 1];
        begin(prog, chain[k], s);
        gl.uniform2i(prog.u.uSrcSize, s.w, s.h);
        gl.uniform1i(prog.u.uFan, chain[k].fan);
        if (prog.u.uChan) gl.uniform1i(prog.u.uChan, k === 0 ? first : rest);
        if (prog.u.uMode) gl.uniform1i(prog.u.uMode, k === 0 ? first : rest);
        tri(prog.name);
      }
    }

    // Motion version 2 (Performance) textures. Sizes: flow grid fw x fh, block grid bw x bh (= the static mask's 8x8 cells), full W x H.
    function allocV2() {
      const lv0 = levels[0], bw = lv0.bw, bh = lv0.bh;
      const gw = Math.ceil(W / 4), gh = Math.ceil(H / 4), tw = Math.ceil(W / 8), th = Math.ceil(H / 8);
      const v = {};
      v.sRaw = mk('v2.sRaw', 'R32F', FW, FH);          // static-structure agreement, before the neighbour test
      v.s = mk('v2.s', 'R32F', FW, FH);                // static-structure mask S
      v.s8 = mk('v2.s8', 'R32F', bw, bh);              // per 8x8 cell: largest S if enough of the cell is in the mask
      v.vfit = mk('v2.vfit', 'RG32F', bw, bh);         // search: raw finest vectors + equiangular sub-pixel fit
      v.repSad = mk('v2.repSad', 'R32F', 3 * bw, 3 * bh);   // 3x3 candidate tile, reused by the re-pick and both refinements
      v.rep0 = mk('v2.rep0', 'RGBA32F', bw, bh);       // re-pick winner: xy, z raw 16-px SAD, w scene-cut term
      v.rep1 = mk('v2.rep1', 'RGBA32F', bw, bh);       // after the 0.5 px refinement
      v.rep2 = mk('v2.rep2', 'RGBA32F', bw, bh);       // after the 0.25 px refinement (= the library's re-picked field)
      v.vf = mk('v2.vf', 'RG32F', bw, bh);             // cleaned block field
      v.ec = mk('v2.ec', 'RG32F', bw, bh);             // per cell: x = enclosure E, y = overlay-near bits C
      v.em = mk('v2.em', 'R32F', W, H);                // per pixel: the taps' mismatch em
      v.grp = mk('v2.grp', 'RGBA32F', gw, gh);         // per 4x4 pixel group: s0, sb, winner + 1, edge flag
      v.morph = mk('v2.morph', colorFmt(), W, H);      // morphed sub-frame
      v.e = mk('v2.e', 'R16F', W, H);                  // per-pixel mismatch e (integer levels)
      v.morphMrt = mkMrt('v2_morph', [v.morph, v.e]);
      v.tile = mk('v2.tile', 'R32F', tw, th);          // per 8x8 tile: valid pixel count if untrusted
      v.untrRed = mkChain('v2.untrRed', tw, th, 16);
      return v;
    }

    // V2 Quality textures (flow grid fw x fh), allocated the first time Quality runs (about 56 bytes per flow pixel).
    function allocQ() {
      const q = {};
      q.qf = mk('v2.qf', 'RGBA32F', FW, FH);           // per-pixel field xy, static-overlay mask z
      q.claim = mk('v2.qclaim', 'RGBA32I', FW, FH);    // own claim: key, vector bits, prev / curr footprint
      q.ownA = mk('v2.qownA', 'R32F', FW, FH);         // winning claim word per pixel of prev / curr (MIN blend of float bits)
      q.ownC = mk('v2.qownC', 'R32F', FW, FH);
      q.vis = mk('v2.qvis', 'RG32F', FW, FH);          // does prev / curr still show the location's content
      q.cov = mk('v2.qcov', 'RGBA32F', FW, FH);        // coverage cP, cC, z (quantised)
      return q;
    }

    // ---- the learned motion estimator (settings.motion 'learned'; ml-fg export/SPEC.md, export/reference.py) ---------------------
    // The net's weights as one uniform buffer: per level (finest first), layer and 12 output channels (a part) a std140 block of the
    // conv shape (learned/conv.frag.glsl's MlW; conv_h.frag.glsl binds part 0 / 1 as MlW / MlW2): COUT * 9 * CIN mat4 (column c =
    // input channel 4j + c, row r = output channel 12 * part + 4o + r) and
    // COUT bias vec4, each block at a multiple of UNIFORM_BUFFER_OFFSET_ALIGNMENT. Decoded and uploaded once per net and engine.
    function mlNetBuffer(name) {
      if (mlBufs[name]) return mlBufs[name];
      const net = mlNetDef(name);
      const bin = root.atob(net.weights);
      if (bin.length !== 4 * net.count) throw new Error('framegen: net ' + name + ': ' + bin.length + ' weight bytes, want ' + 4 * net.count);
      const dv = new DataView(new ArrayBuffer(bin.length)), u8 = new Uint8Array(dv.buffer);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      const wt = new Float32Array(net.count);
      for (let i = 0; i < net.count; i++) wt[i] = dv.getFloat32(4 * i, true);
      const G = net.width / 4, align = gl.getParameter(gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT);
      const blocks = [], datas = [];
      let total = 0;
      for (let l = 0; l < net.levels; l++) {
        const st = net.stacks[net.levelToStack[l]], lv = [];
        for (let k = 0; k <= net.depth; k++) {
          const ly = st[k], last = k === net.depth;
          if (ly.cin !== (k === 0 ? net.inChannels : net.width) || ly.cout !== (last ? 2 : net.width) || ly.act !== (last ? 'none' : 'clamp01')) {
            throw new Error('framegen: net ' + name + ': unexpected layer ' + k + ' of level ' + l);
          }
          const CIN = k === 0 ? 4 : G, COUT = last ? 1 : 3, parts = [];
          for (let part = 0; part < (last ? 1 : G / 3); part++) {
            const n = COUT * 9 * CIN, f = new Float32Array(n * 16 + COUT * 4);
            for (let o = 0; o < COUT; o++) {
              for (let t = 0; t < 9; t++) {
                for (let j = 0; j < CIN; j++) {
                  const m = ((o * 9 + t) * CIN + j) * 16;
                  for (let c = 0; c < 4; c++) {
                    for (let r = 0; r < 4; r++) {
                      const oc = 12 * part + 4 * o + r, ic = 4 * j + c;   // W[oc][ic][ky][kx] (OIHW), t = ky * 3 + kx
                      if (oc < ly.cout && ic < ly.cin) f[m + 4 * c + r] = wt[ly.w + (oc * ly.cin + ic) * 9 + t];
                    }
                  }
                }
              }
              for (let r = 0; r < 4; r++) { const oc = 12 * part + 4 * o + r; if (oc < ly.cout) f[n * 16 + 4 * o + r] = wt[ly.b + oc]; }
            }
            total = Math.ceil(total / align) * align;
            parts.push({ off: total, size: f.byteLength });
            datas.push([total, f]);
            total += f.byteLength;
          }
          lv.push(parts);
        }
        blocks.push(lv);
      }
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.UNIFORM_BUFFER, buf);
      gl.bufferData(gl.UNIFORM_BUFFER, total, gl.STATIC_DRAW);
      for (const [off, f] of datas) gl.bufferSubData(gl.UNIFORM_BUFFER, off, f);
      gl.bindBuffer(gl.UNIFORM_BUFFER, null);
      if (debug) chk('mlNetBuffer');
      return (mlBufs[name] = { name: name, net: net, buf: buf, blocks: blocks, bytes: total });
    }

    // drop some of the owned textures / FBOs (the learned set on a net switch)
    function freeSome(tex, fbos) {
      for (const t of tex) { gl.deleteFramebuffer(t.fbo); gl.deleteTexture(t.tex); }
      for (const f of fbos) gl.deleteFramebuffer(f);
      ownedTex = ownedTex.filter(function (t) { return tex.indexOf(t) < 0; });
      ownedFbo = ownedFbo.filter(function (f) { return fbos.indexOf(f) < 0; });
      resetCache();
    }

    // The learned estimator's textures (allocated by its first estimate; again when the net or the precision changes) and its pass
    // plan. Sizes: padded flow grid PW x PH (fw, fh rounded up to a multiple of 16), level l = (PW >> l) x (PH >> l), every level an
    // exact half.
    //   pyr[0] RGBA32F (la, lc, ya, yc), pyr[l > 0] RG32F (la, lc)   the net's own pyramid (SPEC stage 3)
    //   x[], level-0 size                                           the 14-channel input of the current level
    //   act[2][], level-0 size                                      ping-pong activations
    //   h[1..4] RG32F, level size                                   the field after each level (level 0 writes pixfield)
    //   dupSrc R32F, dupRed, sadRed                                  the duplicate-test and SAD sum chains
    // motionPrecision 'fp32': x = 3 RGBA32F + 1 RG32F, act = G = width / 4 RGBA32F each; 'fp16': 8 half floats per RGBA32UI texel,
    // x = 2, act = ceil(G / 2) each. The coarser levels render into the level-0-sized x / act textures over their own viewport
    // (uSize = the level: zero padding).
    function allocMl(name, prec) {
      if (T.ml) {
        if (T.ml.name === name && T.ml.prec === prec) return T.ml;
        freeSome(T.ml.tex, T.ml.fbos);
        T.ml = null;
      }
      const half = prec === 'fp16';
      const nb = mlNetBuffer(name), net = nb.net, G = net.width / 4, NL = net.levels;
      const PW = Math.ceil(FW / net.padMultiple) * net.padMultiple, PH = Math.ceil(FH / net.padMultiple) * net.padMultiple;
      const t0 = ownedTex.length, f0 = ownedFbo.length;
      let m = null;
      try {
        m = { name: name, prec: prec, net: net, buf: nb.buf, G: G, PW: PW, PH: PH, pyr: [], h: [null], act: [[], []], actMrt: [[], []], plan: [] };
        m.pyr.push(mk('ml.pyr[0]', 'RGBA32F', PW, PH));
        m.dupSrc = mk('ml.dupSrc', 'R32F', PW, PH);
        m.lvMrt = mkMrt('ml_lv0', [m.pyr[0], m.dupSrc]);
        for (let l = 1; l < NL; l++) {
          m.pyr.push(mk('ml.pyr[' + l + ']', 'RG32F', PW >> l, PH >> l));
          m.h.push(mk('ml.h[' + l + ']', 'RG32F', PW >> l, PH >> l));
        }
        const xf = half ? ['RGBA32UI', 'RGBA32UI'] : ['RGBA32F', 'RGBA32F', 'RGBA32F', 'RG32F'];
        m.x = xf.map(function (f, k) { return mk('ml.x[' + k + ']', f, PW, PH); });
        m.xMrt = mkMrt('ml_match', m.x);
        const nAct = half ? Math.ceil(G / 2) : G;
        for (let s = 0; s < 2; s++) {
          for (let g = 0; g < nAct; g++) m.act[s].push(mk('ml.act[' + s + '][' + g + ']', half ? 'RGBA32UI' : 'RGBA32F', PW, PH));
          if (half) m.actMrt[s].push(mkMrt('ml_conv', m.act[s]));
          else for (let part = 0; part < G / 3; part++) m.actMrt[s].push(mkMrt('ml_conv', m.act[s].slice(3 * part, 3 * part + 3)));
        }
        m.dupRed = mkChain('ml.dupRed', PW, PH, 4, 4);   // (small fan-ins: many fragments, short loops)
        m.sadRed = mkChain('ml.sadRed', FW, FH, 2, 4);
        const dup = m.dupRed[m.dupRed.length - 1];
        const chkBlock = function (p, k, b) {
          if (p.wSize[k] !== b.size) throw new Error('framegen: ' + p.name + ': uniform block ' + ['MlW', 'MlW2'][k] + ' is ' + p.wSize[k] + ' bytes, the host packs ' + b.size);
        };
        // the coarse-to-fine plan: per level the match, then the layers into the ping-pong sets (fp32, G = 6: parts a / b, each
        // with its weight block; fp16: one pass, the parts' blocks bound to MlW / MlW2), the last layer into h[l] (level 0:
        // pixfield, fw x fh; its taps still read the padded level)
        for (let l = NL - 1; l >= 0; l--) {
          const lw = PW >> l, lh = PH >> l, size = [lw, lh], top = l === NL - 1 ? 1 : 0, hc = top ? m.pyr[l] : m.h[l + 1];   // (hc unread at the top)
          m.plan.push({ name: 'ml_match.' + l, p: P[half ? 'ml_match_h' : 'ml_match'], out: { fbo: m.xMrt.fbo, w: lw, h: lh }, size: size, top: top,
            ins: [m.pyr[l], hc, dup], block: null, block2: null });
          for (let k = 0; k <= net.depth; k++) {
            const last = k === net.depth, src = k === 0 ? m.x : m.act[(k - 1) & 1], parts = nb.blocks[l][k];
            const p = P[mlLayerProg(prec, G, k, net.depth)], ins = src.concat(last ? [hc, dup] : [dup]);
            const outOf = function (part) { return !last ? { fbo: m.actMrt[k & 1][part].fbo, w: lw, h: lh } : l === 0 ? T.pixfield : { fbo: m.h[l].fbo, w: lw, h: lh }; };
            if (half) {
              chkBlock(p, 0, parts[0]);
              if (parts[1]) chkBlock(p, 1, parts[1]);
              m.plan.push({ name: 'ml_c' + k + '.' + l, p: p, out: outOf(0), size: size, top: top, ins: ins, block: parts[0], block2: parts[1] || null });
              continue;
            }
            for (let part = 0; part < parts.length; part++) {
              chkBlock(p, 0, parts[part]);
              m.plan.push({ name: 'ml_c' + k + (parts.length > 1 ? 'ab'[part] : '') + '.' + l, p: p, out: outOf(part), size: size, top: top,
                ins: ins, block: parts[part], block2: null });
            }
          }
        }
        if (debug) chk('allocMl');
      } catch (e) {
        freeSome(ownedTex.slice(t0), ownedFbo.slice(f0));
        throw e;
      }
      m.tex = ownedTex.slice(t0);
      m.fbos = ownedFbo.slice(f0);
      T.ml = m;
      return m;
    }

    function alloc(w, h) {
      const sizeErr = fg.frameSizeError(gl, w, h, settings);
      if (sizeErr) throw new RangeError('framegen: ' + sizeErr);   // (before freeing: the engine keeps its current size)
      freeTextures();
      const lay = fg.levelLayout(w, h, settings.flowScale);
      W = w; H = h; FW = lay.fw; FH = lay.fh; levels = lay.levels; L = levels.length;
      const lv0 = levels[0];
      try {
        const t = {};
        t.frame = [mk('frame[0]', colorFmt(), W, H, true), mk('frame[1]', colorFmt(), W, H, true)];
        t.luma = [[], []];
        for (let s = 0; s < 2; s++) {
          for (let i = 0; i < L; i++) t.luma[s].push(mk('luma[' + s + '][' + i + ']', 'R32F', levels[i].w, levels[i].h));
        }
        t.hist = [];   // per level i < L-1: 64 x (2 x bands of MH_ROWS parent rows), 4 bins per texel
        for (let i = 0; i + 1 < L; i++) t.hist.push(mk('hist[' + i + ']', 'RGBA32F', 64, 2 * Math.ceil(levels[i + 1].bh / MH_ROWS)));
        t.median = []; t.seedSad = []; t.seed = []; t.searchSad = []; t.field = []; t.fieldRaw = [];
        for (let i = 0; i < L; i++) {
          const lv = levels[i], D = 2 * lv.radius + 1;
          if (i < L - 1) {
            t.median.push(mk('median[' + i + ']', 'RG32F', 1, 1));
            t.seedSad.push(mk('seedSad[' + i + ']', 'R32F', 6 * lv.bw, lv.bh));
          }
          t.seed.push(mk('seed[' + i + ']', 'RG32F', lv.bw, lv.bh));   // seed[L-1] is never rendered: stays zero
          t.searchSad.push(mk('searchSad[' + i + ']', 'R32F', D * lv.bw, D * lv.bh));
          t.field.push(mk('field[' + i + ']', 'RG32F', lv.bw, lv.bh));
          t.fieldRaw.push(mk('fieldRaw[' + i + ']', 'RG32F', lv.bw, lv.bh));   // the search's argmin before the 3x3 median
        }
        t.subSad = mk('subSad', 'R32F', 3 * lv0.bw, 3 * lv0.bh);
        t.sub1 = mk('sub1', 'RGBA32F', lv0.bw, lv0.bh);
        t.sub2 = mk('sub2', 'RGBA32F', lv0.bw, lv0.bh);
        t.sadRed = mkChain('sadRed', lv0.bw, lv0.bh, 16);
        t.pfErr0 = mk('pfErr0', 'RGBA32F', FW, FH);
        t.pfErr1 = mk('pfErr1', 'RGBA32F', FW, FH);
        t.pfErrMrt = mkMrt('pf_err', [t.pfErr0, t.pfErr1]);
        t.pixfield = mk('pixfield', 'RG32F', FW, FH);
        t.fieldT = mk('fieldT', 'RG32F', FW, FH);
        t.warpA = mk('warpA', colorFmt(), W, H);
        t.warpC = mk('warpC', colorFmt(), W, H);
        t.mismatch = mk('mismatch', 'R8', W, H);
        t.errMrt = mkMrt('error', [t.mismatch, t.warpA, t.warpC]);
        t.hsum = mk('hsum', 'R16UI', W, H);
        t.trust = mk('trust', 'R32F', W, H);
        t.untrRed = mkChain('untrRed', W, H, 8);   // (8x8 first: more fragments over the full-res trust)
        t.out = mk('out', colorFmt(), W, H, true);
        if (settings.motionVersion === 2) t.v2 = allocV2();
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        T = t;
        if (debug) chk('alloc');
      } catch (e) {
        freeTextures();
        throw e;
      }
      info = {
        W: W, H: H, fw: FW, fh: FH, flowScale: settings.flowScale,
        levels: levels,
      };
      engine.info = info;
      staticDirty = true;
      resetCache();
    }

    // ---- uniforms ---------------------------------------------------------------------------
    function applyStatics() {
      const s = settings, nb = levels[0].bw * levels[0].bh;
      const trustScale = Math.fround(1 / (s.trustHigh - s.trustLow));
      for (const p of progList) {
        const u = p.u;
        use(p);
        if (u.uFrameSize) gl.uniform2i(u.uFrameSize, W, H);
        if (u.uFlowScale) gl.uniform1i(u.uFlowScale, s.flowScale);
        if (u.uNumBlocks) gl.uniform1i(u.uNumBlocks, nb);
        if (u.uLambda) gl.uniform1f(u.uLambda, s.searchPenalty);
        if (u.uSmoothBonus) gl.uniform1f(u.uSmoothBonus, s.smoothBonus);
        if (u.uZeroBonus) gl.uniform1f(u.uZeroBonus, s.hudBonus);
        if (u.uCutSad) gl.uniform1f(u.uCutSad, s.sceneCutSad);
        if (u.uTrust0) gl.uniform1f(u.uTrust0, s.trustLow);
        if (u.uTrustScale) gl.uniform1f(u.uTrustScale, trustScale);
        if (u.uGlobalFrac) gl.uniform1f(u.uGlobalFrac, s.globalFallback);
        if (u.uTrustMid) gl.uniform1f(u.uTrustMid, 0.5 * (s.trustLow + s.trustHigh));
      }
      staticDirty = false;
      if (debug) chk('applyStatics');
    }

    function setTime(p) {
      const u = p.u;
      if (u.uT) gl.uniform1f(u.uT, tT);
      if (u.uS) gl.uniform1f(u.uS, tS);
      if (u.uWA) gl.uniform1f(u.uWA, tWA);
      if (u.uWC) gl.uniform1f(u.uWC, tWC);
      if (u.uNearC) gl.uniform1i(u.uNearC, tNear);
      if (u.uMid) gl.uniform1i(u.uMid, tMid);
    }

    // ---- pass plumbing (all allocation-free) ------------------------------------------------
    function use(p) {
      if (curProg !== p) { gl.useProgram(p.prog); curProg = p; }
    }

    function setVp(x, y, w, h) {
      if (vpX !== x || vpY !== y || vpW !== w || vpH !== h) {
        gl.viewport(x, y, w, h);
        vpX = x; vpY = y; vpW = w; vpH = h;
      }
    }

    function bindFbo(fbo) {
      if (curFbo !== fbo) { gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); curFbo = fbo; }
    }

    function bindIn(unit, t) {
      if (units[unit] !== t.tex) {
        if (curUnit !== unit) { gl.activeTexture(gl.TEXTURE0 + unit); curUnit = unit; }
        gl.bindTexture(gl.TEXTURE_2D, t.tex);
        units[unit] = t.tex;
      }
    }

    // out: { fbo, w, h } ; inputs bound to units 0.. in order
    function begin(p, out) {
      use(p);
      if (p.hasTime) setTime(p);
      bindFbo(out.fbo);
      setVp(0, 0, out.w, out.h);
      for (let k = 2; k < arguments.length; k++) if (arguments[k]) bindIn(k - 2, arguments[k]);
    }

    // begin() with the inputs as an array (the learned plan)
    function beginArr(p, out, ins) {
      use(p);
      if (p.hasTime) setTime(p);
      bindFbo(out.fbo);
      setVp(0, 0, out.w, out.h);
      for (let k = 0; k < ins.length; k++) bindIn(k, ins[k]);
    }

    let passHook = null;   // engine.setPassHook (profiling only): called before / after every draw
    function draw(name, mode, n) {
      if (passHook) passHook(name, 0);
      gl.drawArrays(mode, 0, n);
      if (passHook) passHook(name, 1);
    }
    function tri(name) {
      draw(name, gl.TRIANGLES, 3);
      nDraw++;
      if (debug) chk(name);
    }

    // Y7 / presentFrame blit. fb: framebuffer or null. Viewport values already resolved.
    function blit(src, fb, x, y, w, h, flip) {
      const p = P.blit;
      use(p);
      bindFbo(fb);
      setVp(x, y, w, h);
      bindIn(0, src);
      gl.uniform2f(p.u.uVpOrigin, x, y);
      gl.uniform2f(p.u.uVpSize, w, h);
      gl.uniform1i(p.u.uFlipY, flip ? 1 : 0);
      gl.uniform1i(p.u.uExact, (w === W && h === H) ? 1 : 0);
      tri('blit');
    }

    // ---- public API -------------------------------------------------------------------------
    const engine = {
      settings: settings,
      info: null,
      support: sup,   // checkSupport(gl) at creation (quality / qualityError)
    };

    engine.resize = function (w, h) {
      alive();
      if (!(w >= 1 && h >= 1) || Math.floor(w) !== w || Math.floor(h) !== h) {
        throw new RangeError('framegen: resize: bad size ' + w + 'x' + h);
      }
      alloc(w, h);
    };

    engine.setSettings = function (partial) {
      alive();
      const next = learnedGate(qualityGate(mergeSettings(settings, partial)));
      const fmtChanged = formatDefines(next) !== formatDefines(settings);
      const rebuilt = (next.flowScale !== settings.flowScale || fmtChanged) && info !== null;
      const switched = next.motionVersion !== settings.motionVersion;
      const estSwitched = estKey(next) !== estKey(settings);   // (motion / motionNet with V1 too)
      if (fmtChanged || (switched && next.motionVersion === 2) || (mlOn(next) && !mlBuilt)) {
        const keep = settings;
        settings = next;   // (srcOf / compile errors surface here, not on the hot path)
        try {
          if (fmtChanged) rebuildPrograms();
          if (next.motionVersion === 2) ensureV2Programs();
          if (mlOn(next)) ensureMlPrograms();
        } catch (e) { settings = keep; if (fmtChanged) rebuildPrograms(); throw e; }
      }
      settings = next;
      debug = settings.debug;
      engine.settings = settings;
      staticDirty = true;
      if (rebuilt) alloc(W, H);   // re-allocates all flow-res (or, for a format change, all) textures, drops frame history
      else if (switched || estSwitched) {
        // like the library: a motion-version switch drops the current estimate (synth returns false until the next estimate()); so
        // does a switch of V1's estimator (motion / motionNet)
        estimatedPair = false; haveEstimate = false; synthDone = false; synthMid = false; lastT = null; mlEst = false;
        if (info && settings.motionVersion === 2 && !T.v2) T.v2 = allocV2();
      }
      return { rebuilt: rebuilt };
    };

    function sourceSize(src) {
      if (src.data !== undefined && src.videoWidth === undefined) return [src.width, src.height];
      if (typeof src.videoWidth === 'number') return [src.videoWidth, src.videoHeight];
      if (typeof src.displayWidth === 'number') return [src.displayWidth, src.displayHeight];   // VideoFrame
      return [src.width, src.height];
    }

    engine.pushFrame = function (source) {
      alive();
      if (!source) throw new TypeError('framegen: pushFrame: no source');
      const raw = source.data !== undefined && source.videoWidth === undefined;
      const sz = sourceSize(source);
      const w = sz[0], h = sz[1];
      if (!(w >= 1 && h >= 1)) throw new Error('framegen: pushFrame: source has no size yet (' + w + 'x' + h + ')');
      if (raw && source.data.length !== 4 * w * h) {
        throw new RangeError('framegen: pushFrame: data.length ' + source.data.length + ' != 4*' + w + '*' + h);
      }
      if (!info || info.W !== w || info.H !== h) alloc(w, h);
      beginCall();
      nDraw = 0;
      const n = 1 - currS;
      const fr = T.frame[n];

      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      bindIn(0, fr);
      const f16 = settings.inputFormat === 'rgba16f';
      if (raw) {
        // rgba8: Uint8Array / Uint8ClampedArray levels; rgba16f: Float32Array values or Uint16Array half-float bits
        const d = source.data;
        const type = !f16 ? gl.UNSIGNED_BYTE : d instanceof Float32Array ? gl.FLOAT : d instanceof Uint16Array ? gl.HALF_FLOAT : 0;
        if (!type || (!f16 && !(d instanceof Uint8Array || d instanceof Uint8ClampedArray))) {
          throw new TypeError('framegen: pushFrame: data must be ' + (f16 ? 'a Float32Array or Uint16Array (half floats)' : 'a Uint8Array') + ' for inputFormat ' + settings.inputFormat);
        }
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, type, d);
      } else {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, f16 ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, source);
      }
      if (debug) chk('upload');

      const lm = T.luma[n];
      begin(P.luma, lm[0], fr);                                  // P1
      tri('luma');
      for (let i = 1; i < L; i++) {                              // P2.i
        begin(P.down, lm[i], lm[i - 1]);
        gl.uniform2i(P.down.u.uSrcSize, levels[i - 1].w, levels[i - 1].h);
        tri('down');
      }

      prevS = currS; currS = n;
      count++;
      estimatedPair = false;
      synthDone = false;
      counts.pushFrame = nDraw;
      return count;
    };

    // V2 static-structure mask (v2_static.comp): S at flow resolution, per-cell maxima. Reads only the level-0 luma.
    function estimateV2Mask(a0, c0) {
      const v = T.v2;
      begin(P.v2_static_raw, v.sRaw, a0, c0);
      tri('v2_static_raw');
      begin(P.v2_static_s, v.s, v.sRaw);
      gl.uniform1i(P.v2_static_s.u.uStaticOff, settings.v2StaticMask ? 0 : 1);
      tri('v2_static_s');
      begin(P.v2_static_cell, v.s8, v.s);
      tri('v2_static_cell');
    }

    // V2 Performance field (v2_repick.comp, v2_clean.comp, v2_enclose.comp) from the search's fitted finest field `vfit`.
    function estimateV2Field(a0, c0) {
      const v = T.v2;
      begin(P.v2_repick_sad, v.repSad, a0, c0, v.vfit);
      tri('v2_repick_sad');
      begin(P.v2_repick_pick, v.rep0, v.repSad, v.vfit);
      tri('v2_repick_pick');
      begin(P.v2_refine_sad, v.repSad, a0, c0, v.rep0);
      gl.uniform1f(P.v2_refine_sad.u.uStep, 0.5);
      tri('v2_refine_sad');
      begin(P.v2_refine_pick, v.rep1, v.repSad, v.rep0);
      gl.uniform1f(P.v2_refine_pick.u.uStep, 0.5);
      tri('v2_refine_pick');
      begin(P.v2_refine_sad, v.repSad, a0, c0, v.rep1);
      gl.uniform1f(P.v2_refine_sad.u.uStep, 0.25);
      tri('v2_refine_sad');
      begin(P.v2_refine_pick, v.rep2, v.repSad, v.rep1);
      gl.uniform1f(P.v2_refine_pick.u.uStep, 0.25);
      tri('v2_refine_pick');
      // scene-cut sum: the final per-block roundEven(256 * mean SAD) / 256 (w), reduced into sadRed[last] (= sadSum)
      runChain(P.reduce_sad, T.sadRed, v.rep2, 3, 0);
      begin(P.v2_clean, v.vf, v.rep2);
      tri('v2_clean');
      begin(P.v2_enclose, v.ec, v.s8, v.vf);
      tri('v2_enclose');
    }

    // V2 Quality's estimate tail (v2_qfield, v2_qown, v2_qcov of the library) from the cleaned block field; once per estimate, run by
    // estimate() in Quality or by the first Quality synth() of an estimate made in Performance (a mode switch keeps the estimate).
    function estimateQ() {
      const v = T.v2;
      if (!v.q) v.q = allocQ();
      const q = v.q, A = T.luma[prevS][0], C = T.luma[currS][0];
      begin(P.v2_qfield, q.qf, A, C, v.vf);
      tri('v2_qfield');
      begin(P.v2_qclaim, q.claim, A, C, q.qf);
      tri('v2_qclaim');
      // the splat (the library's atomicMin of key << 20 | vector): per side, the static-overlay claims initialise the target
      // (v2_qown_z), then one MIN-blended point per location (v2/qown.vert.glsl: the words as float bits)
      const po = P.v2_qown, pz = P.v2_qown_z, n = FW * FH;
      for (let side = 0; side < 2; side++) {
        const dst = side === 0 ? q.ownA : q.ownC;
        begin(pz, dst, q.qf);
        tri('v2_qown_z');
        use(po);
        bindIn(0, q.claim);
        gl.uniform1i(po.u.uSide, side);
        gl.enable(gl.BLEND);
        gl.blendEquation(gl.MIN);
        draw('v2_qown', gl.POINTS, n);
        gl.disable(gl.BLEND);
        gl.blendEquation(gl.FUNC_ADD);
        nDraw++;
        if (debug) chk('v2_qown');
      }
      begin(P.v2_qvis, q.vis, q.claim, q.ownA, q.ownC);
      tri('v2_qvis');
      begin(P.v2_qcov, q.cov, q.vis, q.qf);
      tri('v2_qcov');
      qDone = true;
    }

    // V2 synthesis at time t (uT set by the caller): group choice, morph, flick count, final. Quality: the same passes with its
    // per-pixel field and coverage.
    function synthV2() {
      const v = T.v2, pf = T.frame[prevS], cf = T.frame[currS];
      if (settings.v2Mode === 'quality') {
        if (!qDone) estimateQ();
        const q = v.q;
        begin(P.v2_qem, v.em, pf, cf, v.vf, v.s, v.ec, q.qf, q.cov);
        tri('v2_qem');
        begin(P.v2_qgroup, v.grp, pf, cf, v.vf, v.s, v.ec, q.qf, q.cov, v.em);
        tri('v2_qgroup');
        begin(P.v2_qmorph, v.morphMrt, pf, cf, v.vf, v.s, v.ec, q.qf, q.cov, v.em, v.grp);
        tri('v2_qmorph');
      } else {
        begin(P.v2_em, v.em, pf, cf, v.vf, v.s, v.ec);
        tri('v2_em');
        begin(P.v2_group, v.grp, pf, cf, v.vf, v.s, v.ec, v.em);
        tri('v2_group');
        begin(P.v2_morph, v.morphMrt, pf, cf, v.vf, v.s, v.ec, v.em, v.grp);
        tri('v2_morph');
      }
      begin(P.v2_tile, v.tile, v.e);
      tri('v2_tile');
      const ur = v.untrRed;
      runChain(P.reduce_untrusted, ur, v.tile, 0, 0);
      begin(P.v2_final, T.out, pf, cf, v.morph, T.sadRed[T.sadRed.length - 1], ur[ur.length - 1]);
      tri('v2_final');
    }

    // a reduction chain whose first step `first` (already bound by the caller's begin) has drawn: the rest with reduce_sad
    function sumRest(chain) {
      const p = P.reduce_sad;
      for (let k = 1; k < chain.length; k++) {
        begin(p, chain[k], chain[k - 1]);
        gl.uniform2i(p.u.uSrcSize, chain[k - 1].w, chain[k - 1].h);
        gl.uniform1i(p.u.uFan, chain[k].fan);
        gl.uniform1i(p.u.uChan, 0);
        tri('reduce_sad');
      }
    }

    // The learned estimate (ml-fg export/SPEC.md stages 1-6): the duplicate test, the net's pyramid, coarse to fine (match, 5 convs
    // per level), then V1's contract: the field in pixfield (fw x fh, flow-res half-vectors) and sadSum = sad * bw * bh in sadRed's
    // last texel. V1's synthesis reads both unchanged (cut: sad > sceneCutSad).
    function estimateLearned() {
      const m = allocMl(settings.motionNet, settings.motionPrecision), A = T.frame[prevS], C = T.frame[currS];
      const dr = m.dupRed, dup = dr[dr.length - 1];
      begin(P.ml_lv0, m.lvMrt, A, C);   // level 0 of the pyramid + the duplicate test's per-cell sums
      gl.uniform2i(P.ml_lv0.u.uSize, FW, FH);
      tri('ml_lv0');
      runChain(P.reduce_sad, dr, m.dupSrc, 0, 0);
      for (let l = 1; l < m.pyr.length; l++) {
        begin(P.ml_down, m.pyr[l], m.pyr[l - 1]);
        tri('ml_down');
      }
      for (let i = 0; i < m.plan.length; i++) {
        const d = m.plan[i], u = d.p.u;
        beginArr(d.p, d.out, d.ins);
        gl.uniform2i(u.uSize, d.size[0], d.size[1]);
        if (u.uTop) gl.uniform1i(u.uTop, d.top);
        if (d.block) gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, m.buf, d.block.off, d.block.size);
        if (d.block2) gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, m.buf, d.block2.off, d.block2.size);
        tri(d.name);
      }
      const sr = m.sadRed;
      begin(P.ml_sad, sr[0], m.pyr[0], T.pixfield);
      gl.uniform1i(P.ml_sad.u.uFan, sr[0].fan);
      tri('ml_sad');
      sumRest(sr);
      const lv0 = levels[0];
      begin(P.ml_sadout, T.sadRed[T.sadRed.length - 1], sr[sr.length - 1], dup);
      gl.uniform1f(P.ml_sadout.u.uWeight, Math.fround(lv0.bw * lv0.bh / (FW * FH)));
      tri('ml_sadout');
    }

    engine.estimate = function () {
      alive();
      if (!info || count < 2) return false;
      beginCall();
      nDraw = 0; nClear = 0;
      mlEst = mlOn(settings);
      if (mlEst) {
        estimateLearned();
        estimatedPair = true;
        haveEstimate = true;
        synthDone = false;
        counts.estimate = nDraw;
        counts.clears = nClear;
        return true;
      }
      const A = T.luma[prevS], C = T.luma[currS];
      const v2 = settings.motionVersion === 2;
      if (v2) {
        if (!T.v2) T.v2 = allocV2();
        estimateV2Mask(A[0], C[0]);
      }

      for (let i = L - 1; i >= 0; i--) {
        const lv = levels[i];
        if (i < L - 1) {
          const par = T.field[i + 1], hist = T.hist[i], med = T.median[i], ss = T.seedSad[i];
          // E2 medhist: the banded histogram of the parent vectors (a gather; every texel written)
          begin(P.medhist, hist, par);
          gl.uniform2i(P.medhist.u.uBlocks, lv.bw, lv.bh);
          tri('medhist');
          // E3 medsel
          begin(P.medsel, med, hist);
          gl.uniform1i(P.medsel.u.uCount, lv.bw * lv.bh);
          tri('medsel');
          // E4 seed_sad
          begin(P.seed_sad, ss, A[i], C[i], par, med);
          tri('seed_sad');
          // E5 seed_pick
          begin(P.seed_pick, T.seed[i], ss, par, med);
          tri('seed_pick');
        }
        // E6 search_sad
        begin(P.search_sad, T.searchSad[i], A[i], C[i], T.seed[i]);
        gl.uniform1i(P.search_sad.u.uRadius, lv.radius);
        tri('search_sad');
        // E7 search_pick (V2, finest level: the raw argmin + equiangular sub-pixel fit, no 3x3 median)
        if (v2 && i === 0) {
          begin(P.v2_search_pick0, T.v2.vfit, T.searchSad[i], T.seed[i]);
          gl.uniform1i(P.v2_search_pick0.u.uRadius, lv.radius);
          tri('v2_search_pick0');
        } else {
          begin(P.search_pick, T.fieldRaw[i], T.searchSad[i], T.seed[i]);
          gl.uniform1i(P.search_pick.u.uRadius, lv.radius);
          tri('search_pick');
          begin(P.median3, T.field[i], T.fieldRaw[i]);
          tri('median3');
        }
      }

      if (v2) {
        estimateV2Field(A[0], C[0]);
        qDone = false;
        if (settings.v2Mode === 'quality') estimateQ();
        estimatedPair = true;
        haveEstimate = true;
        synthDone = false;
        counts.estimate = nDraw;
        counts.clears = nClear;
        return true;
      }

      // E8..E11 sub-pixel refinement (0.5 px then 0.25 px)
      begin(P.subpix_sad, T.subSad, A[0], C[0], T.field[0]);
      gl.uniform1f(P.subpix_sad.u.uStep, 0.5);
      tri('subpix_sad');
      begin(P.subpix_pick, T.sub1, T.subSad, T.field[0]);
      gl.uniform1f(P.subpix_pick.u.uStep, 0.5);
      tri('subpix_pick');
      begin(P.subpix_sad, T.subSad, A[0], C[0], T.sub1);
      gl.uniform1f(P.subpix_sad.u.uStep, 0.25);
      tri('subpix_sad');
      begin(P.subpix_pick, T.sub2, T.subSad, T.sub1);
      gl.uniform1f(P.subpix_pick.u.uStep, 0.25);
      tri('subpix_pick');

      // E12 SAD reduction chain (last = sadSum)
      runChain(P.reduce_sad, T.sadRed, T.sub2, 2, 0);

      // E13/E14 pixel field
      begin(P.pf_err, T.pfErrMrt, A[0], C[0], T.sub2);
      tri('pf_err');
      begin(P.pf_select, T.pixfield, T.pfErr0, T.pfErr1, T.sub2);
      tri('pf_select');

      estimatedPair = true;
      haveEstimate = true;
      synthDone = false;
      counts.estimate = nDraw;
      counts.clears = nClear;
      return true;
    };

    // the time uniforms of t, then the generated frame at t into T.out (V1 / V2, with the whole-frame fallbacks)
    function synthOut(t) {
      const mid = t === 0.5;
      tT = Math.fround(t);
      tS = Math.fround(2 * t - 1);
      tWA = Math.fround(1 - t);
      tWC = Math.fround(t);
      tNear = t > 0.5 ? 1 : 0;
      tMid = mid ? 1 : 0;
      if (settings.motionVersion === 2) {
        synthV2();   // group choice, morph, flick count, final (the whole-frame fallbacks only)
      } else {
        // Y1 lookup (t != 0.5)
        let field = T.pixfield;
        if (!mid) {
          begin(P.lookup, T.fieldT, T.pixfield);
          tri('lookup');
          field = T.fieldT;
        }
        // Y2 error + warps (MRT)
        begin(P.error, T.errMrt, T.frame[prevS], T.frame[currS], field);
        tri('error');
        // Y3 / Y4 box filters
        begin(P.hbox, T.hsum, T.mismatch);
        tri('hbox');
        begin(P.vbox, T.trust, T.hsum);
        tri('vbox');
        // Y5 untrusted count reduction
        const ur = T.untrRed;
        runChain(P.reduce_untrusted, ur, T.trust, 1, 0);
        // Y6 final
        begin(P.final, T.out, T.frame[prevS], T.frame[currS], T.warpA, T.warpC, T.trust,
          T.sadRed[T.sadRed.length - 1], ur[ur.length - 1]);
        tri('final');
      }
      synthDone = true;
      synthMid = mid;
      lastT = t;
    }

    // Y7: T.out to the caller's framebuffer (undefined: none, the out FBO stays bound)
    function present(framebuffer, viewport, opts) {
      if (framebuffer !== undefined) {
        const isNull = framebuffer === null;
        let x = 0, y = 0, w = W, h = H;
        if (viewport) { x = viewport.x; y = viewport.y; w = viewport.width; h = viewport.height; }
        else if (isNull) { w = gl.drawingBufferWidth; h = gl.drawingBufferHeight; }
        const flip = opts && opts.flipY !== undefined ? !!opts.flipY : isNull;
        blit(T.out, framebuffer, x, y, w, h, flip);
      } else {
        bindFbo(T.out.fbo);
        setVp(0, 0, W, H);
      }
    }

    engine.synth = function (t, framebuffer, viewport, opts) {
      alive();
      if (!(t > 0 && t < 1)) throw new RangeError('framegen: synth: t must be in (0,1), got ' + t);
      if (!info || !estimatedPair) return false;
      beginCall();
      nDraw = 0;
      synthOut(t);
      present(framebuffer, viewport, opts);
      counts.synth = nDraw;
      return true;
    };

    // Motion blur (the library's recordSynthBlur; V2 only here): one output = the weighted average of settings.blurSamples (N)
    // sub-frames at the midpoints of N equal slices of [t0, t1], t_k = t0 + (t1 - t0)(k + 0.5)/N, weights settings.blurWeights
    // ('gaussian': exp(-2 x_k^2), x_k = (2k + 1 - N)/N; 'even': 1/N). Each sub-frame is exactly synth(t_k)'s output (its own
    // scene-cut / flick fallback); a sample at t <= 0 / t >= 1 is the real frame prev / curr. Blur off (N < 2) or t0 == t1: the
    // instant t1 (synth(t1); t1 = 0 / 1: a copy of prev / curr). 0 <= t0 <= t1 <= 1, else false. V1 with blur on throws (the
    // library's V1 blur, blur_*.comp, is a different algorithm and is not ported). stats() after it: the last sample's counts.
    engine.synthBlur = function (t0, t1, framebuffer, viewport, opts) {
      alive();
      if (!(t0 >= 0 && t1 <= 1 && t0 <= t1)) return false;
      const N = settings.blurSamples, blur = N >= 2 && t0 < t1;
      if (!blur && t1 > 0 && t1 < 1) return engine.synth(t1, framebuffer, viewport, opts);
      if (!info || !estimatedPair) return false;
      if (blur && settings.motionVersion !== 2) throw new Error('framegen: synthBlur: motion blur is implemented for motionVersion 2 only');
      beginCall();
      nDraw = 0;
      const n = blur ? N : 1, gauss = settings.blurWeights === 'gaussian';
      if (!T.blurAcc) T.blurAcc = mk('blurAcc', 'RGBA32F', W, H);
      let ws = 0;
      for (let k = 0; k < n; k++) {
        const t = blur ? Math.min(Math.max(Math.fround(t0 + (t1 - t0) * (k + 0.5) / n), t0), t1) : t1;
        const x = Math.fround((2 * k + 1 - n) / n);
        const w = Math.fround(gauss ? Math.exp(-2 * x * x) : 1);
        ws = Math.fround(ws + w);
        let src = T.out;
        if (t <= 0) src = T.frame[prevS];
        else if (t >= 1) src = T.frame[currS];
        else synthOut(t);
        begin(P.blur_accum, T.blurAcc, src);
        gl.uniform1f(P.blur_accum.u.uWeight, w);
        if (k > 0) { gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ONE, gl.ONE); }
        tri('blur_accum');
        gl.disable(gl.BLEND);
      }
      begin(P.blur_resolve, T.out, T.blurAcc);
      gl.uniform1f(P.blur_resolve.u.uWsum, ws);
      tri('blur_resolve');
      synthDone = true;
      present(framebuffer, viewport, opts);
      counts.synth = nDraw;
      return true;
    };

    engine.presentFrame = function (which, framebuffer, viewport, opts) {
      alive();
      if (which !== 'prev' && which !== 'curr') throw new TypeError("framegen: presentFrame: which must be 'prev' or 'curr'");
      if (!info || count < (which === 'prev' ? 2 : 1)) return false;
      beginCall();
      nDraw = 0;
      if (framebuffer === undefined) framebuffer = null;   // no "out only" meaning here: default framebuffer
      const isNull = framebuffer === null;
      let x = 0, y = 0, w = W, h = H;
      if (viewport) { x = viewport.x; y = viewport.y; w = viewport.width; h = viewport.height; }
      else if (isNull) { w = gl.drawingBufferWidth; h = gl.drawingBufferHeight; }
      const flip = opts && opts.flipY !== undefined ? !!opts.flipY : isNull;
      blit(T.frame[which === 'prev' ? prevS : currS], framebuffer, x, y, w, h, flip);
      return true;
    };

    // Profiling / test only: fn(passName, 0 before | 1 after) around every draw call (null = off). Not for the hot path.
    engine.setPassHook = function (fn) { passHook = typeof fn === 'function' ? fn : null; };

    engine.passCounts = function () {
      return { pushFrame: counts.pushFrame, estimate: counts.estimate, synth: counts.synth, clears: counts.clears };
    };

    // ---- debug readback ---------------------------------------------------------------------
    function readTex(t) {
      const f = t.fmt;
      const buf = new f.A(t.w * t.h * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
      gl.readPixels(0, 0, t.w, t.h, f.rf, f.rt, buf);
      const err = gl.getError();
      resetCache();
      if (err !== gl.NO_ERROR) throw new Error('framegen: readPixels(' + t.name + ') GL error 0x' + err.toString(16));
      if (f.ch === 4) return buf;
      const out = new f.A(t.w * t.h * f.ch);
      for (let i = 0, n = t.w * t.h; i < n; i++) {
        for (let c = 0; c < f.ch; c++) out[i * f.ch + c] = buf[i * 4 + c];
      }
      return out;
    }

    function level(parts, idx, limit, name) {
      const i = Number(parts[idx]);
      if (!(Number.isInteger(i) && i >= 0 && i < limit)) throw new Error('framegen: debugRead: bad level in "' + name + '"');
      return i;
    }

    function needEstimate(name) {
      if (!haveEstimate) throw new Error('framegen: debugRead("' + name + '"): no estimate yet');
    }
    function needSynth(name) {
      if (!synthDone) throw new Error('framegen: debugRead("' + name + '"): no synth yet');
    }

    function resolve(name) {
      if (!info) throw new Error('framegen: debugRead("' + name + '"): engine has no frames yet');
      const parts = String(name).split('.');
      switch (parts[0]) {
        case 'frame': {
          const w = parts[1];
          if (w !== 'prev' && w !== 'curr') break;
          if (count < (w === 'prev' ? 2 : 1)) throw new Error('framegen: debugRead("' + name + '"): frame not pushed yet');
          return T.frame[w === 'prev' ? prevS : currS];
        }
        case 'luma': {
          const w = parts[1];
          if (w !== 'prev' && w !== 'curr') break;
          if (count < (w === 'prev' ? 2 : 1)) throw new Error('framegen: debugRead("' + name + '"): frame not pushed yet');
          return T.luma[w === 'prev' ? prevS : currS][level(parts, 2, L, name)];
        }
        case 'hist': needEstimate(name); return T.hist[level(parts, 1, L - 1, name)];
        case 'fieldRaw': needEstimate(name); return T.fieldRaw[level(parts, 1, L, name)];
        case 'median': needEstimate(name); return T.median[level(parts, 1, L - 1, name)];
        case 'seedSad': needEstimate(name); return T.seedSad[level(parts, 1, L - 1, name)];
        case 'seed': needEstimate(name); return T.seed[level(parts, 1, L, name)];
        case 'searchSad': needEstimate(name); return T.searchSad[level(parts, 1, L, name)];
        case 'field': needEstimate(name); return T.field[level(parts, 1, L, name)];
        case 'subSad': needEstimate(name); return T.subSad;
        case 'sub1': needEstimate(name); return T.sub1;
        case 'sub2': case 'blockflow': needEstimate(name); return T.sub2;
        case 'sadSum': needEstimate(name); return T.sadRed[T.sadRed.length - 1];
        case 'pfErr0': needEstimate(name); return T.pfErr0;
        case 'pfErr1': needEstimate(name); return T.pfErr1;
        case 'pixfield': needEstimate(name); return T.pixfield;
        case 'fieldT':
          needSynth(name);
          if (synthMid) throw new Error('framegen: debugRead("fieldT"): last synth was t=0.5 (no fieldT)');
          return T.fieldT;
        case 'warpA': needSynth(name); return T.warpA;
        case 'warpC': needSynth(name); return T.warpC;
        case 'mismatch': needSynth(name); return T.mismatch;
        case 'hsum': needSynth(name); return T.hsum;
        case 'trust': needSynth(name); return T.trust;
        case 'untrusted': {
          needSynth(name);
          const ur = settings.motionVersion === 2 ? T.v2.untrRed : T.untrRed;
          return ur[ur.length - 1];
        }
        case 'v2': {
          // V2 stages: v2.sRaw v2.s v2.s8 v2.vfit v2.rep0 v2.rep1 v2.rep2 v2.vf v2.ec (estimate), v2.em v2.grp v2.morph v2.e v2.tile (synth)
          if (!T.v2) throw new Error('framegen: debugRead("' + name + '"): no V2 textures (motionVersion is 1)');
          const key = parts[1];
          const est = ['sRaw', 's', 's8', 'vfit', 'rep0', 'rep1', 'rep2', 'vf', 'ec'];
          if (est.indexOf(key) >= 0) { needEstimate(name); return T.v2[key]; }
          if (['em', 'grp', 'morph', 'e', 'tile'].indexOf(key) >= 0) { needSynth(name); return T.v2[key]; }
          // Quality: v2.qf v2.qclaim v2.qownA v2.qownC v2.qvis v2.qcov (its estimate tail)
          const qk = { qf: 'qf', qclaim: 'claim', qownA: 'ownA', qownC: 'ownC', qvis: 'vis', qcov: 'cov' }[key];
          if (qk) {
            needEstimate(name);
            if (!qDone) throw new Error('framegen: debugRead("' + name + '"): no Quality estimate tail for this estimate yet');
            return T.v2.q[qk];
          }
          break;
        }
        case 'ml': {
          // the learned estimator (its last estimate): ml.pyr.<l>, ml.x.<k>, ml.act.<set>.<k>, ml.h.<l> (l >= 1), ml.dup (1x1 sum of
          // |a - c|), ml.sad (1x1 sum of the SAD terms). x / act are level-0 sized: after an estimate they hold level 0
          if (!T.ml) throw new Error('framegen: debugRead("' + name + '"): no learned estimate yet');
          const m = T.ml, key = parts[1];
          if (key === 'pyr') return m.pyr[level(parts, 2, m.pyr.length, name)];
          if (key === 'x') return m.x[level(parts, 2, m.x.length, name)];
          if (key === 'act') return m.act[level(parts, 2, 2, name)][level(parts, 3, m.act[0].length, name)];
          if (key === 'h') { const l = level(parts, 2, m.h.length, name); if (l > 0) return m.h[l]; break; }
          if (key === 'dup') return m.dupRed[m.dupRed.length - 1];
          if (key === 'sad') return m.sadRed[m.sadRed.length - 1];
          break;
        }
        case 'out': needSynth(name); return T.out;
        default: break;
      }
      throw new Error('framegen: debugRead: unknown texture "' + name + '"');
    }

    engine.debugRead = function (name) {
      alive();
      const t = resolve(name);
      const data = readTex(t);
      return { name: name, width: t.w, height: t.h, channels: t.fmt.ch, data: data };
    };

    engine.stats = function () {
      alive();
      if (!info || !haveEstimate) throw new Error('framegen: stats: no estimate yet');
      const lv0 = levels[0];
      const sadSum = readTex(T.sadRed[T.sadRed.length - 1])[0];
      const meanSad = sadSum / (lv0.bw * lv0.bh);
      const cut = meanSad > settings.sceneCutSad;
      let untrusted = null, share = null, flick = null, fallback = null, duplicate = null, dupDiff = null;
      if (mlEst && T.ml) {   // the learned estimate: the duplicate test (mean |a - c| over pixels and R, G, B, in levels)
        const dr = T.ml.dupRed;
        dupDiff = readTex(dr[dr.length - 1])[0] / (3 * W * H);
        duplicate = dupDiff <= ML_DUP_DIFF;
      }
      if (synthDone) {
        const ur = settings.motionVersion === 2 ? T.v2.untrRed : T.untrRed;
        untrusted = readTex(ur[ur.length - 1])[0];
        share = untrusted / (W * H);
        flick = untrusted > settings.globalFallback * W * H;
        fallback = cut ? 'cut' : (flick ? 'untrusted' : null);
      }
      return {
        meanSad: meanSad, cut: cut, untrusted: untrusted, untrustedShare: share, flick: flick,
        fallback: fallback, t: synthDone ? lastT : null, duplicate: duplicate, dupDiff: dupDiff,
      };
    };

    engine.dispose = function () {
      if (disposed) return;
      freeTextures();
      for (const p of progList) gl.deleteProgram(p.prog);
      for (const k of Object.keys(mlBufs)) gl.deleteBuffer(mlBufs[k].buf);
      gl.deleteVertexArray(vao);
      progList.length = 0;
      disposed = true;
    };

    return engine;
  };
})(globalThis);
