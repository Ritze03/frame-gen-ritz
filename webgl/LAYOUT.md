# WebGL2 backend: V1 pass contract

The pass-level contract of V1 in the WebGL2 backend (`webgl/`): files, textures, programs, sequences, exactness rules and the host
API. It began as the browser extension's design contract for its port of V1 and is maintained here with the backend. Motion version
2 (Performance and Quality), the frame formats (RGBA16F), the motion blur and the tests are in `README.md`; where the two differ,
README.md is newer. Changes against the extension's original contract are marked **(changed)**; `Dn` and "the scout" refer to the
decisions and the feasibility study of the extension's original plan.

**Reference** (pinned at frame-gen-ritz commit `b77d058`; cites below are `file:line` at that commit):
`proto/interp.py` (NumPy ground truth, abbreviated **PY**), `gpu/shaders/*.comp` (abbreviated by name), `gpu/shaders/common.glsl` (**RC**), `gpu/framegen.cpp` (**FG**), `gpu/framegen.h` (**FH**).
Core path = RGBA8, `Settings` defaults of `FH:68-212`; RGBA16F input follows `format.glsl`'s conventions (README "Frame formats"). UI protection (`ui_*`, CrosshairV2) and V1's motion blur (`blur_*`) are not ported. `subpixel` (default `true`) is **always on** in this port.

---

## 1. File layout

```
webgl/
  README.md, LAYOUT.md               design docs
  framegen.js                        host. classic script, attaches globalThis.__fg
  nets/e2v5.js, nets/e1v5.js,        the learned estimator's weights (classic scripts, attach __fg.nets[<name>]; §10);
  nets/e2v4.js, nets/e1v4.js,        vendor all six (e2v5 is the default net, e1v5 the small one: __fg.smallNet)
  nets/e2v3.js, nets/e1v3.js
  shaders/
    common.glsl                      begins with `#version 300 es`
    vert.glsl                        fullscreen-triangle vertex shader, standalone (own #version)
    estimate/
      luma.frag.glsl                 down.frag.glsl
      medhist.frag.glsl              medsel.frag.glsl                                   (changed: no medhist.vert.glsl)
      seed_sad.frag.glsl             seed_pick.frag.glsl
      search_sad.frag.glsl           search_pick.frag.glsl    median3.frag.glsl         (changed: median3 split out)
      subpix_sad.frag.glsl           subpix_pick.frag.glsl
      reduce_sad.frag.glsl
      pf_err.frag.glsl               pf_select.frag.glsl
    synth/
      lookup.frag.glsl               error.frag.glsl
      hbox.frag.glsl                 vbox.frag.glsl
      reduce_untrusted.frag.glsl     final.frag.glsl          blit.frag.glsl
    v2/                              motion version 2 (README.md)
    learned/                         the learned motion estimator (§10)
      ml_common.inc.glsl             include: duplicate test mlDup, bilinear mlBil, field upsample mlFieldUp
      lv0.frag.glsl                  down.frag.glsl           match.frag.glsl (+ ML_PACKED)
      conv.frag.glsl (fp32)          conv_h.frag.glsl (fp16, packed)
      sad.frag.glsl                  sadout.frag.glsl
      zt_cost.frag.glsl              zt.frag.glsl             the zero-motion test (§10.8)
      sguard.inc.glsl                include: the static guard's constants (one marked block) and weight (§10.9)
      sg_h.frag.glsl                 sg_v.frag.glsl           the static guard's filters (+ synth/final.frag.glsl FG_SGUARD)
```

One file = one program (`vert.glsl` is shared by every V1 program; V2 Quality's splat `v2/qown` has its own vertex shader). Some V2 programs share a fragment file with different `#define`s (README.md); `__fg.assemblePrograms(sources, settings)` returns the exact text the host compiles.

### 1.1 Source assembly rules (host + validate.sh + harness all do the same)

* **`*.frag.glsl` files contain no `#version`, no `precision`, no `#include`.** The host compiles `common.glsl + "\n" + defines + "#line 1\n" + [include + "\n#line 1\n"] + file`, with the frame-format defines inserted after `common.glsl`'s first line (`#version 300 es`; none for RGBA8). (changed)
* **`vert.glsl` and `v2/qown.vert.glsl` are standalone** (own `#version 300 es` + `precision highp float; precision highp int; precision highp sampler2D;`); `common.glsl` is **not** prepended to vertex shaders.
* `common.glsl` declares **no uniforms, no `out`s, no `main`**. Every `.frag.glsl` declares its own uniforms/samplers/outputs, named exactly as in §3.
* `tests/validate.mjs` runs `glslangValidator --stdin -S frag|vert` on exactly the assembled text (`__fg.assemblePrograms`), for every frame-format variant. (changed)
* `framegen.js` never fetches: `sources` keys are the paths relative to `shaders/` (§6), listed in `__fg.shaderFiles`.

### 1.2 `common.glsl` contract (what it must provide)

Header: `#version 300 es`, `precision highp float; precision highp int; precision highp sampler2D; precision highp usampler2D;`.

Constants: `const int BLOCK = 8;`

All sampler arguments are read with `texelFetch(t, ·, 0)`; every "size" is `textureSize(t, 0)` (all engine textures are allocated at exactly the logical size, so no size parameters). **Position arguments `p` are in pixel-index coordinates: integer = pixel centre (cv2 convention), no +0.5, never normalised.**

| Helper (exact signature) | Mirrors | Behaviour |
|---|---|---|
| `float fgRound(float x)` / `vec3 fgRound(vec3 x)` | `FG_FETCH` RC/format.glsl:49 | `floor(x + 0.5)` |
| `float fgFetchL(sampler2D t, ivec2 p)` | `L()` RC:55 | `texelFetch(t, clamp(p, 0, size-1), 0).r` (clamp-to-edge = `BORDER_REPLICATE` = `pad_to_block`) |
| `vec2 fgFetchF(sampler2D t, ivec2 p)` | `F()` RC:72 | same, `.rg` |
| `float fgSampleL(sampler2D t, vec2 p)` | `sampleL()` RC:61-69 | `X = ivec2(roundEven(p*32.0)); i = X>>5; f = vec2(X&31)/32;` 4 clamped `fgFetchL` taps, weights `(1-f.y)(1-f.x), (1-f.y)f.x, f.y(1-f.x), f.y f.x` summed in that order. = `cv2.remap` INTER_LINEAR float, PY:92 |
| `vec2 fgSampleF(sampler2D t, vec2 p)` | `sampleF()` RC:75-83 | same on `.rg` via `fgFetchF` (PY:204) |
| `vec3 fgSampleRGB8(sampler2D t, vec2 p)` | `sampleRGBRaw`+`FG_FETCH` RC:102, `mismatch` RC:136 | same 1/32 quantisation and replicate clamp as `fgSampleL`, but taps are RGBA8 texels converted to integer levels `fgRound(texelFetch(...).rgb*255.0)`, weights as above, result `floor(sum + 0.5)` → **integer-valued vec3 in 0..255** (= `cv2.remap` on uint8: PY:92,214). Manual 4 taps, never `texture()`. |
| `vec2 fgHalfVec(sampler2D field, ivec2 p, int S)` | `halfVec()` RC:86-97 | full-res pixel `p` → half-vector in **full-res pixels**: `fs = textureSize(field,0); p = min(p, S*fs-1); s = (vec2(p)+0.5)/float(S) - 0.5; fl = floor(s); f = s-fl;` taps at `clamp(ivec2(fl) [+1], 0, fs-1)`; `return float(S) * mix(mix(v00,v01,f.x), mix(v10,v11,f.x), f.y)`. = `cv2.resize(pf, ×S, INTER_LINEAR)` + edge pad `* FLOW_SCALE` (PY:229-231). No 1/32 quantisation here (cv2.resize has none). |
| `vec2 fgBlockV(sampler2D bf, ivec2 b)` | `blockV()` pixfield.comp:13 | block-field fetch, `b` clamped to `textureSize(bf,0)-1`, `.xy` |
| `vec2 fgPixCand(int k, ivec2 q, sampler2D bf)` | `cand()` pixfield.comp:18-34, PY:172-176 | per-pixel candidate vector k∈0..6 at flow pixel `q`: 0 bilinear block-field (`s=(q+.5)/8-.5`, two `mix` levels exactly as pixfield.comp:20-27), 1 own block `q/8`, 2 above `b+(0,-1)`, 3 below `(0,+1)`, 4 left `(-1,0)`, 5 right `(+1,0)`, 6 `vec2(0)`. `bf` is read `.xy` (it is `sub2`, RGBA32F). |
| `ivec2 fgSeedCand(int k, ivec2 blk, sampler2D parent, sampler2D median)` | search.comp:25-34, PY:131-143 | seed candidate k∈0..5 for fine block `blk`: `pb = textureSize(parent,0); px=min(blk.x/2,pb.x-1), py=min(blk.y/2,pb.y-1); sy=(blk.y&1)==0?-1:1; sx=(blk.x&1)==0?-1:1; pp = {(0,0),(0,sy),(sx,0),(sx,sy)}`; k<4: `2*ivec2(roundEven(fetch(parent, clamp(ivec2(px,py)+pp[k],0,pb-1)).xy))`; k=4: `ivec2(0)`; k=5: `ivec2(roundEven(texelFetch(median,ivec2(0),0).xy))` |
| `uint fgLuma8(uvec3 rgb)` | luma.comp:24, `cv2 BGR2GRAY` PY:65 | `(rgb.r*4899u + rgb.g*9617u + rgb.b*1868u + 8192u) >> 14` |
| `float fgMed9(float v[9])` | median3.comp:7-13 | median of 9 |

Adding helpers is free, renaming/removing the listed ones is not. Format-aware helpers (`fgPx`, `fgTap`, `fgLumaPx`, `fgStore`, `fgPackW`/`fgUnpackW`, `FG_D`/`FG_G`/`FG_LV`) are described in README "Frame formats"; the colour passes use them, so with RGBA8 they reduce to the helpers above. `fgMed9` is a 19-exchange sorting network (same result). (changed)

---

## 2. Texture table

Notation: `W×H` = source frame size (set by `resize`); `S` = `flowScale` ∈ {1,2,4}; `rhe(x)` = round-half-to-even (= `cv2.saturate_cast<int>` = `FG:346 rintHalfEven`); flow size `fw = rhe(W/S), fh = rhe(H/S)`.
**Pyramid levels** `i = 0..L-1` (0 = finest = flow res): `(w0,h0) = (fw,fh)`; `(w_{i+1},h_{i+1}) = (rhe(w_i/2), rhe(h_i/2))`; level `i+1` exists iff `floor(min(w_i,h_i)/2) >= 48` (FG:373-386, PY:73-77 `MIN_SIDE`). Block grid `bw_i = ceil(w_i/8), bh_i = ceil(h_i/8)` (= `pad_to_block`). Search radius `R_i = 6` for `i = L-1` (TOP_RADIUS), else `2` (RADIUS); `D_i = 2R_i+1`.
Example sizes at 1920×1080: S=1 → L=5 (1920×1080, 960×540, 480×270, 240×135, 120×68); S=2 → L=4 (960×540 … 120×68); S=4 → L=3 (480×270, 240×135, 120×68).

All textures: `texStorage2D`, `TEXTURE_WRAP_S/T = CLAMP_TO_EDGE`, filter `NEAREST` (min and mag, no mips) **except** RGBA8 `frame[s]`/`out` which are `LINEAR` (only the blit's `texture()` path uses it; every other pass reads them with `texelFetch`, which ignores filtering). WebGL zero-initialises texStorage, which the contract uses for `seed[L-1]`.
Formats are written `internalformat / format / type`. Rows: **row 0 = top image row** (§5.1).

Lifetime key: **slot** = lives with a frame slot (2 slots, ping-pong: `pushFrame` writes slot `1-curr`, then `prev := old curr, curr := new`); **pair** = overwritten by the next `estimate()`; **synth** = overwritten by the next `synth()`; **static** = created at `resize`, constant.

| Name (debugRead key) | Format `internal / format / type` | Size | Life | Written by → read by |
|---|---|---|---|---|
| `frame[s]`, s∈{0,1}  (`frame.prev`, `frame.curr`) | `RGBA8 / RGBA / UNSIGNED_BYTE`, LINEAR | W×H | slot | upload → `luma`, `final`, `blit` |
| `luma[s][i]` (`luma.prev.i`, `luma.curr.i`) | `R32F / RED / FLOAT` | w_i×h_i | slot | `luma` (i=0), `down` (i>0) → seed_sad, search_sad, subpix_sad, pf_err |
| `hist[i]`, i<L-1 (changed) | `RGBA32F` | 64×(2·B_i), `B_i = ceil(bh_{i+1}/4)` bands of 4 parent rows; texel `(g, c·B+band)`.j = cumulative weight of bins ≤ 4g+j of component c in that band | pair | `medhist` (gather) → `medsel` |
| `median[i]`, i<L-1 | `RG32F / RG / FLOAT` | 1×1 | pair | `medsel` → seed_sad, seed_pick |
| `seedSad[i]`, i<L-1 | `R32F` | (6·bw_i)×bh_i, x = 6·bx + k | pair | `seed_sad` → `seed_pick` |
| `seed[i]` all i<L; **`seed[L-1]` is never rendered to and stays all-zero** | `RG32F` | bw_i×bh_i | pair | `seed_pick` → search_sad, search_pick |
| `searchSad[i]` | `R32F` | (D_i·bw_i)×(D_i·bh_i); texel `(D·bx+cx, D·by+cy)` = candidate offset `(cx−R, cy−R)` of block `(bx,by)` | pair | `search_sad` → `search_pick` |
| `fieldRaw[i]` (changed: new) | `RG32F` | bw_i×bh_i | pair | `search_pick` (argmin) → `median3` |
| `field[i]` (post-median3 block vectors; integer-valued for i≥1) | `RG32F` | bw_i×bh_i | pair | `median3` → next level's medhist/seed_*, `subpix_*` (i=0) |
| `subSad` | `R32F` | (3·bw_0)×(3·bh_0); texel `(3bx+cx, 3by+cy)` = offset `(cx−1,cy−1)·step` | pair (reused by both sub-pixel passes) | `subpix_sad` → `subpix_pick` |
| `sub1` | `RGBA32F` | bw_0×bh_0; xy = vector after the 0.5 px pass | pair | `subpix_pick` → `subpix_sad` (2nd pass) |
| `sub2` (alias `blockflow`) | `RGBA32F` | bw_0×bh_0; **xy = final block half-vector = `flow()` v (PY:164)**, z = winning mean SAD of the block (= `flow()` per-block `sad`), w = 0 | pair | `subpix_pick` → reduce_sad, pf_err, pf_select |
| `sadRed[k]`, k=0..K−1 (last = **`sadSum`**, 1×1) | `R32F` | `s_0 = ceil(bw_0/16)×ceil(bh_0/16)`, `s_{k+1} = ceil(s_k/16)`, until 1×1 (≥1 step even if already 1×1). `sadSum` = Σ block mean-SAD | pair | `reduce_sad` → `final` |
| `pfErr0` | `RGBA32F` | fw×fh; (e0,e1,e2,e3) | pair | `pf_err` → `pf_select` |
| `pfErr1` | `RGBA32F` | fw×fh; (e4,e5,e6,0) | pair | `pf_err` → `pf_select` |
| `pixfield` | `RG32F` | fw×fh | pair | `pf_select` → `lookup` / `error` (= `pixel_field()` PY:167) |
| `fieldT` (only when t≠0.5) | `RG32F` | fw×fh | synth | `lookup` → `error` (= `field_at()` PY:194) |
| `warpA` | `RGBA8` | W×H; integer levels /255, a=1 | synth | `error` → `final` |
| `warpC` | `RGBA8` | W×H | synth | `error` → `final` |
| `mismatch` | `R8 / RED / UNSIGNED_BYTE` | W×H; value `e/255`, e = max-channel \|wa−wc\| ∈ 0..255 integer | synth | `error` → `hbox` |
| `hsum` | `R16UI / RED_INTEGER / UNSIGNED_SHORT` | W×H; Σ of 7 horizontal `e` (≤1785) | synth | `hbox` → `vbox` |
| `trust` | `R32F` | W×H; t = ramp(7×7 mean of e) | synth | `vbox` → `reduce_untrusted`, `final` |
| `untrRed[k]` (last = **`untrusted`**, 1×1) | `R32F` | changed: `ceil(prev/8)` first, then `ceil(prev/16)` until 1×1 (1080p: 240×135, 15×9, 1×1) | synth | `reduce_untrusted` → `final` |
| `out` | `RGBA8` (LINEAR) | W×H | synth | `final` → `blit` / debug |

Not allocated: no mip levels, no depth/stencil, no renderbuffers. Max texture dimension used ≈ `D·bw_0` (5·480 = 2400 at 4K, S=1) — far below `MAX_TEXTURE_SIZE`. `MAX_DRAW_BUFFERS` needed = 3 (the scout's "8" is not required; ES 3.0 guarantees 4).

---

## 3. Programs and passes

### 3.1 Conventions (all passes)

* Every pass: bind FBO whose attachments are the listed outputs (`COLOR_ATTACHMENT0..n`, `drawBuffers` set to match), `gl.viewport(0,0,vw,vh)`, bind program, bind input textures, set uniforms, `drawArrays(TRIANGLES, 0, 3)` with `vert.glsl` (except `medhist`: `drawArrays(POINTS, 0, n)`). Viewport origin is always (0,0) for internal passes, so **`ivec2(gl_FragCoord.xy)` is the target texel index** (row 0 = texture row 0). Fragments need no varyings from the vertex shader.
* **Sampler units**: in each program the samplers are bound to units `0,1,2,…` in the order listed in the program table (host does `uniform1i(name, unit)` once at link). The shader must declare exactly those names/types.
* **Automatic (global) uniforms**: the host sets, for *every* program that declares one of these names, the value below (type must match). Per-pass uniforms (§3.4) are set per pass.

| Uniform | GLSL | Value / source |
|---|---|---|
| `uFrameSize` | `ivec2` | (W, H) |
| `uFlowScale` | `int` | S (`settings.flowScale`) |
| `uNumBlocks` | `int` | `bw_0 * bh_0` |
| `uLambda` | `float` | `settings.searchPenalty` (LAMBDA, PY:48) |
| `uSmoothBonus` | `float` | `settings.smoothBonus` (PY:52) |
| `uZeroBonus` | `float` | `settings.hudBonus` (PY:53 ZERO_BONUS) |
| `uCutSad` | `float` | `settings.sceneCutSad` (PY:50) |
| `uTrust0` | `float` | `settings.trustLow` |
| `uTrustScale` | `float` | `Math.fround(1/(trustHigh−trustLow))` (RC:46: folded on CPU) |
| `uGlobalFrac` | `float` | `settings.globalFallback` |
| `uT` | `float` | t of the current `synth` call (`fround`) |
| `uS` | `float` | `fround(2t−1)` (computed in double first, FG:1740-1745) |
| `uWA`, `uWC` | `float` | `fround(1−t)`, `fround(t)` (double first). Warp factors are `2·uWC` (a side) and `2·uWA` (c side) |
| `uNearC` | `int` | `t > 0.5 ? 1 : 0` (near frame = prev for t ≤ 0.5, PY:223) |
| `uMid` | `int` | `t == 0.5 ? 1 : 0` |

### 3.2 Program table

`out` declarations are verbatim; render targets of format R32F/RG32F/RGBA32F/R8/RGBA8 take `vec4` outputs (unused channels ignored by the format), R16UI takes `uvec4`. "Job" cites the reference.

**Estimate**

| Program (file) | Samplers (unit order) | Own uniforms | Outputs | Job |
|---|---|---|---|---|
| `luma` (`estimate/luma.frag.glsl`) | `sampler2D uFrame` | `uFrameSize`, `uFlowScale` | `layout(location=0) out vec4 oLuma;` | flow-res luma: `o=ivec2(gl_FragCoord.xy)`; for j then i in 0..S−1: `p=S*o+(i,j)`, if `p.x<W && p.y<H`: `c=uvec3(texelFetch(uFrame,p,0).rgb*255.0+0.5)`, `sum+=float(fgLuma8(c))`, `n++`; `oLuma=vec4(sum/float(n))`. luma.comp:22-25,61-70; PY:150 (`cvtColor` BGR2GRAY then INTER_AREA 1/S). |
| `down` (`estimate/down.frag.glsl`) | `sampler2D uSrc` | `ivec2 uSrcSize` | `layout(location=0) out vec4 oLuma;` | pyramid: mean of the *valid* pixels of the 2×2 cell: `p=2*o+(i,j)`, valid if `p.x<uSrcSize.x && p.y<uSrcSize.y`; `s = n==4 ? 0.25 : 1.0/float(n)`; `oLuma=vec4(sum*s)`. down.comp:11-24; PY:76. |
| `medhist` (`estimate/medhist.frag.glsl`) (changed) | `sampler2D uParent` | `ivec2 uBlocks` (fine level bw,bh) | `layout(location=0) out vec4 oCount;` | **gather** (§3.5): per texel `(g, c·B+band)` the cumulative weight of the parents of its band whose component c bins (`clamp(roundEven(v)+128,0,255)`) to ≤ 4g+j; a parent weighs the number of fine blocks mapping to it (`min(ix/2, pw-1)`). medhist.comp:24-30. |
| `medsel` (`estimate/medsel.frag.glsl`) (changed) | `sampler2D uHist` | `int uCount` (= bw·bh of the fine level) | `layout(location=0) out vec4 oMedian;` | for comp c∈{0,1}: `k1=uint(uCount-1)/2u, k2=uint(uCount)/2u`; `kth(k)` = the first bin b whose cumulative count (Σ of the B bands) is `> k`, found by binary search (else 127), as `float(b-128)`; `m=0.5*(kth(k1)+kth(k2))`; write **`2.0*m`**: `oMedian=vec4(2m_x, 2m_y, 0, 0)`. medhist.comp:10-37; PY:142. |
| `seed_sad` (`estimate/seed_sad.frag.glsl`) | `uLumaA`, `uLumaC` (level i luma of prev/curr), `uParent` (= `field[i+1]`), `uMedian` (= `median[i]`) — all `sampler2D` | – | `layout(location=0) out vec4 oSad;` | `f=ivec2(gl_FragCoord.xy); blk=ivec2(f.x/6,f.y); k=f.x%6; cand=fgSeedCand(k,blk,uParent,uMedian)`; `oSad.r` = Σ over 64 px (j=0..63, y-outer/x-inner, `q=blk*8+ivec2(j&7,j>>3)`) of `abs(fgFetchL(uLumaA,q-cand) - fgFetchL(uLumaC,q+cand))` — **raw sum, not mean**. search.comp:35-37; PY:143 (`block_sad`). |
| `seed_pick` (`estimate/seed_pick.frag.glsl`) | `uSeedSad`, `uParent`, `uMedian` | `uLambda` | `layout(location=0) out vec4 oSeed;` | `best=sum0*(1.0/64.0), bi=0`; for k=1..5: `c=sum_k*(1.0/64.0)+uLambda; if(c<best){best=c;bi=k;}` (**strict <, k ascending = first wins**); `oSeed=vec4(vec2(fgSeedCand(bi,…)),0,0)`. search.comp:44-52; PY:143-144. |
| `search_sad` (`estimate/search_sad.frag.glsl`) | `uLumaA`, `uLumaC`, `uSeed` | `int uRadius` (R_i) | `layout(location=0) out vec4 oSad;` | `D=2*uRadius+1; f=ivec2(gl_FragCoord.xy); blk=f/D; dd=f%D-uRadius; seed=ivec2(roundEven(texelFetch(uSeed,blk,0).xy)); cand=seed+dd`; `oSad.r` = 64-px raw SAD sum as in `seed_sad` for `cand`. (At level L−1 `uSeed`=`seed[L-1]`=all zero ⇒ seed 0.) search.comp:57-71; PY:116-120. |
| `search_pick` (`estimate/search_pick.frag.glsl`) (changed) | `uSearchSad`, `uSeed` | `uLambda`, `int uRadius` | `layout(location=0) out vec4 oField;` | **argmin** of the block: seed as above; scan `ci=0..D²−1` row-major (`dy` outer, `dx` inner, `d=(ci%D−R, ci/D−R)`): `cost = sum*(1.0/64.0) + uLambda*float(abs(d.x)+abs(d.y))`, **strict <**; `oField = vec4(seed + d_best, 0, 0)` → `fieldRaw[i]`. search.comp:63-79; PY:121-127. |
| `median3` (`estimate/median3.frag.glsl`) (new) | `uRaw` (= `fieldRaw[i]`) | – | `layout(location=0) out vec4 oField;` | 3×3 per-component median, replicate border: `vec4(fgMed9(x of the 9), fgMed9(y of the 9), 0, 0)` → `field[i]`. median3.comp:15-27; PY:128. |
| `subpix_sad` (`estimate/subpix_sad.frag.glsl`) | `uLumaA`, `uLumaC` (level 0), `uBase` | `float uStep` | `layout(location=0) out vec4 oSad;` | `f=ivec2(gl_FragCoord.xy); blk=f/3; d=vec2(f%3-1); h = texelFetch(uBase,blk,0).xy + d*uStep`; for y=0..7: `row=Σ_{x=0..7} abs(fgSampleL(uLumaA, vec2(q)-h) - fgSampleL(uLumaC, vec2(q)+h))`, `q=blk*8+ivec2(x,y)`; `sum+=row` (rows summed after, as subpix.comp:44-53). `oSad.r=sum` (raw). subpix.comp:38-56; PY:162-163. |
| `subpix_pick` (`estimate/subpix_pick.frag.glsl`) | `uSubSad`, `uBase` | `uLambda`, `float uStep` | `layout(location=0) out vec4 oSub;` | scan `ci=0..8` (`d=(ci%3−1, ci/3−1)`, `dy` outer): `cost = sum*(1.0/64.0) + uLambda*uStep*float(abs(d.x)+abs(d.y))`, strict <. `oSub = vec4(base + vec2(d_best)*uStep, sum_best*(1.0/64.0), 0)`. subpix.comp:50-65. |
| `reduce_sad` (`estimate/reduce_sad.frag.glsl`) | `sampler2D uSrc` | `ivec2 uSrcSize`, `int uChan`, `int uFan` (changed) | `layout(location=0) out vec4 oSum;` | sum of the (≤uFan×uFan) block of source texels `uFan*o + (i,j)` inside `uSrcSize`, component `uChan`; `oSum=vec4(sum)`. Replaces the u32 fixed-point atomic of subpix.comp:67 / fallback.comp:9 (§5.2). |
| `pf_err` (`estimate/pf_err.frag.glsl`) | `uLumaA`, `uLumaC` (level 0), `uBlockField` (= `sub2`) | – | `layout(location=0) out vec4 oErr0; layout(location=1) out vec4 oErr1;` | `q=ivec2(gl_FragCoord.xy)`; for k=0..6: `v=fgPixCand(k,q,uBlockField); e_k=abs(fgSampleL(uLumaA,vec2(q)-v) - fgSampleL(uLumaC,vec2(q)+v))`; `oErr0=vec4(e0,e1,e2,e3)`, `oErr1=vec4(e4,e5,e6,0)`. (Each window pixel uses **its own** candidate vector, as `cv2.blur` over the per-candidate error image does.) pixfield.comp:41-45; PY:177-180. |
| `pf_select` (`estimate/pf_select.frag.glsl`) | `uPfErr0`, `uPfErr1`, `uBlockField` | `uSmoothBonus`, `uZeroBonus` | `layout(location=0) out vec4 oField;` | `q=ivec2(gl_FragCoord.xy)`, `fs=textureSize(uPfErr0,0)`; for k=0..6: `sum=Σ_{y=-2..2}Σ_{x=-2..2} err_k(clamp(q+(x,y),0,fs-1))` (y-outer); `c=sum*(1.0/25.0)`; `if(k==0) c-=uSmoothBonus; if(k==6) c-=uZeroBonus;` `if(k==0 || c<best){best=c;bk=k;}` (strict <, first wins). `oField=vec4(fgPixCand(bk,q,uBlockField),0,0)`. pixfield.comp:47-61; PY:181-183. |

**Synth**

| Program (file) | Samplers (unit order) | Own uniforms | Outputs | Job |
|---|---|---|---|---|
| `lookup` (`synth/lookup.frag.glsl`) | `sampler2D uField` (= `pixfield`) | `uS` | `layout(location=0) out vec4 oField;` | `q=ivec2(gl_FragCoord.xy); g=texelFetch(uField,q,0).xy;` twice: `g=fgSampleF(uField, vec2(q) - uS*g);` `oField=vec4(g,0,0)`. LOOKUP_ITERS=2. lookup.comp:12-17; PY:194-205. Not run when `uMid` (s==0). |
| `error` (`synth/error.frag.glsl`) | `uPrev`, `uCurr` (RGBA8 frames), `uField` (= `pixfield` if t==0.5 else `fieldT`) | `uFlowScale`, `uWA`, `uWC` | `layout(location=0) out vec4 oE; layout(location=1) out vec4 oWarpA; layout(location=2) out vec4 oWarpC;` | `p=ivec2(gl_FragCoord.xy); hv=fgHalfVec(uField,p,uFlowScale); ka=2.0*uWC; kc=2.0*uWA; wa=fgSampleRGB8(uPrev, vec2(p)-ka*hv); wc=fgSampleRGB8(uCurr, vec2(p)+kc*hv); e=max(abs(wa-wc))` (max over r,g,b; integer-valued); `oE=vec4(e/255.0,0,0,1); oWarpA=vec4(wa/255.0,1); oWarpC=vec4(wc/255.0,1)`. error.comp:9-20, RC:130-142; PY:229-236. |
| `hbox` (`synth/hbox.frag.glsl`) | `sampler2D uE` (= `mismatch`) | `uFrameSize` | `layout(location=0) out uvec4 oH;` | `s=Σ_{k=-3..3} uint(floor(texelFetch(uE, ivec2(clamp(x+k,0,W-1), y), 0).r*255.0+0.5))`; `oH=uvec4(s,0u,0u,0u)`. resolve.comp:27-31; PY:236. |
| `vbox` (`synth/vbox.frag.glsl`) | `usampler2D uH` (= `hsum`) | `uFrameSize`, `uTrust0`, `uTrustScale` | `layout(location=0) out vec4 oTrust;` | `s=Σ_{k=-3..3} texelFetch(uH, ivec2(x, clamp(y+k,0,H-1)),0).r` (uint, exact); `e=float(s)*(1.0/49.0); t=clamp((e-uTrust0)*uTrustScale,0.0,1.0); oTrust=vec4(t,0,0,1)`. resolve.comp:38-42; PY:236-237. (The ">0.5" flag is taken in `reduce_untrusted`'s first step, not stored.) |
| `reduce_untrusted` (`synth/reduce_untrusted.frag.glsl`) | `sampler2D uSrc` | `ivec2 uSrcSize`, `int uMode` (1 = first step: value is `texel.r > 0.5 ? 1.0 : 0.0`; 0 = value is `texel.r`), `int uFan` (changed) | `layout(location=0) out vec4 oSum;` | sum over the (≤uFan×uFan) block `uFan*o+(i,j)` inside `uSrcSize`. Final 1×1 = count of pixels with t > 0.5 (resolve.comp:43; PY:238 `(u>0.5).mean()`). |
| `final` (`synth/final.frag.glsl`) | `uPrev`, `uCurr`, `uWarpA`, `uWarpC` (RGBA8), `uTrust` (R32F), `uSadSum`, `uUntrusted` (1×1 R32F) | `uFrameSize`, `uNumBlocks`, `uCutSad`, `uGlobalFrac`, `uWA`, `uWC`, `uNearC`, `uMid` | `layout(location=0) out vec4 oColor;` | `p=ivec2(gl_FragCoord.xy); near = uNearC!=0 ? texelFetch(uCurr,p,0) : texelFetch(uPrev,p,0)`. `cut = texelFetch(uSadSum,(0,0),0).r / float(uNumBlocks) > uCutSad`; `flick = texelFetch(uUntrusted,(0,0),0).r > uGlobalFrac*float(W)*float(H)`. If `cut\|\|flick`: `oColor = vec4(near.rgb, 1.0)`. Else `wa,wc = fgRound(texelFetch(uWarpA/uWarpC,p,0).rgb*255.0)`, `t=texelFetch(uTrust,p,0).r`, `m = uMid!=0 ? 0.5*(wa+wc) : uWA*wa + uWC*wc`, `n8=fgRound(near.rgb*255.0)`, `o=m*(1.0-t)+n8*t`, `oColor=vec4(clamp(floor(o+0.5),0.0,255.0)/255.0, 1.0)`. fallback.comp:9-17; resolve.comp:72-81; PY:224-246 (TIME_BLEND). |
| `blit` (`synth/blit.frag.glsl`) | `sampler2D uSrc` (RGBA8) | `vec2 uVpOrigin`, `vec2 uVpSize` (target viewport in pixels), `int uFlipY`, `int uExact` | `layout(location=0) out vec4 oColor;` | `c = gl_FragCoord.xy - uVpOrigin` (pixel index in viewport + 0.5); `if(uFlipY!=0) c.y = uVpSize.y - c.y;` `uExact!=0` (viewport size == frame size): `oColor=vec4(texelFetch(uSrc, ivec2(c), 0).rgb,1)` (exact copy); else `oColor=vec4(texture(uSrc, c/uVpSize).rgb,1)` (bilinear, LINEAR filter). Presentation only; not in the reference. |

### 3.3 Sequences

`prevS`/`currS` = slot indices of the older/newer frame. Draw = one `drawArrays`. "A/C" = `uLumaA`/`uLumaC` ← `luma[prevS][i]`/`luma[currS][i]`.

**`pushFrame(source)`** (slot `n = 1−currS`):

| # | Program | Inputs (unit: texture) | Output | Viewport | Per-pass uniforms |
|---|---|---|---|---|---|
| U | *(upload, no draw)* | – | `frame[n]` via `texSubImage2D` (§6) | – | – |
| P1 | `luma` | 0 `uFrame`: `frame[n]` | `luma[n][0]` | fw×fh | – |
| P2.i, i=1..L−1 | `down` | 0 `uSrc`: `luma[n][i−1]` | `luma[n][i]` | w_i×h_i | `uSrcSize = (w_{i−1}, h_{i−1})` |

Then `prev:=curr slot`, `curr:=n`. The pyramid of the previous frame is *reused* (built when that frame was pushed): one pyramid per real frame.

**`estimate()`** on (prevS, currS) — requires two pushed frames of the current size:

| # | Program | Inputs (unit: texture) | Output | Viewport | Per-pass uniforms |
|---|---|---|---|---|---|
| for i = L−1 … 0: | | | | | |
| E2.i (i<L−1) | `medhist` (changed: gather, no clear, no blend) | 0 `uParent`: `field[i+1]` | `hist[i]` | 64×2B_i | `uBlocks=(bw_i,bh_i)` |
| E3.i (i<L−1) | `medsel` | 0 `uHist`: `hist[i]` | `median[i]` | 1×1 | `uCount = bw_i·bh_i` |
| E4.i (i<L−1) | `seed_sad` | 0 A, 1 C, 2 `field[i+1]`, 3 `median[i]` | `seedSad[i]` | (6·bw_i)×bh_i | – |
| E5.i (i<L−1) | `seed_pick` | 0 `seedSad[i]`, 1 `field[i+1]`, 2 `median[i]` | `seed[i]` | bw_i×bh_i | – |
| E6.i | `search_sad` | 0 A, 1 C, 2 `seed[i]` | `searchSad[i]` | (D_i·bw_i)×(D_i·bh_i) | `uRadius=R_i` |
| E7.i | `search_pick` | 0 `searchSad[i]`, 1 `seed[i]` | `fieldRaw[i]` | bw_i×bh_i | `uRadius=R_i` |
| E7b.i (changed: new) | `median3` | 0 `fieldRaw[i]` | `field[i]` | bw_i×bh_i | – |
| E8 | `subpix_sad` | 0 A₀, 1 C₀, 2 `field[0]` | `subSad` | 3bw_0×3bh_0 | `uStep=0.5` |
| E9 | `subpix_pick` | 0 `subSad`, 1 `field[0]` | `sub1` | bw_0×bh_0 | `uStep=0.5` |
| E10 | `subpix_sad` | 0 A₀, 1 C₀, 2 `sub1` | `subSad` | 3bw_0×3bh_0 | `uStep=0.25` |
| E11 | `subpix_pick` | 0 `subSad`, 1 `sub1` | `sub2` | bw_0×bh_0 | `uStep=0.25` |
| E12.k, k=0..K−1 | `reduce_sad` | 0 `uSrc`: k=0 → `sub2`, else `sadRed[k−1]` | `sadRed[k]` (last = `sadSum`) | `s_k` | `uSrcSize` = size of the source texture; `uChan` = 2 for k=0, else 0 |
| E13 | `pf_err` | 0 A₀, 1 C₀, 2 `sub2` | `pfErr0` (att 0), `pfErr1` (att 1) | fw×fh | – |
| E14 | `pf_select` | 0 `pfErr0`, 1 `pfErr1`, 2 `sub2` | `pixfield` | fw×fh | – |

The pixel field is always computed (no GPU-side skip on a scene cut; PY:190 only saves CPU work). `synth` honours the cut via `final`.

**`synth(t, framebuffer, viewport)`**, t ∈ (0,1), after `estimate()` and before the next `pushFrame`:

| # | Program | Inputs (unit: texture) | Output | Viewport | Per-pass uniforms |
|---|---|---|---|---|---|
| Y1 (t≠0.5 only) | `lookup` | 0 `pixfield` | `fieldT` | fw×fh | – (`uS` global) |
| Y2 | `error` | 0 `frame[prevS]`, 1 `frame[currS]`, 2 `fieldT` (t≠0.5) or `pixfield` (t=0.5) | `mismatch` (att 0), `warpA` (att 1), `warpC` (att 2) | W×H | – |
| Y3 | `hbox` | 0 `mismatch` | `hsum` | W×H | – |
| Y4 | `vbox` | 0 `hsum` | `trust` | W×H | – |
| Y5.k | `reduce_untrusted` | 0: k=0 → `trust`, else `untrRed[k−1]` | `untrRed[k]` (last = `untrusted`) | `u_k` (`u_0 = ceil(W/16)×ceil(H/16)`) | `uSrcSize`; `uMode` = 1 for k=0 else 0 |
| Y6 | `final` | 0 `frame[prevS]`, 1 `frame[currS]`, 2 `warpA`, 3 `warpC`, 4 `trust`, 5 `sadSum`, 6 `untrusted` | `out` | W×H | – |
| Y7 (only if `framebuffer !== undefined`) | `blit` | 0 `out` | `framebuffer` (`null` = canvas) | `viewport` | `uVpOrigin=(vx,vy)`, `uVpSize=(vw,vh)`, `uFlipY = (framebuffer===null)`, `uExact = (vw==W && vh==H)` |

`error` attachments are fixed: **att0 = `mismatch` (oE), att1 = `warpA` (oWarpA), att2 = `warpC` (oWarpC)**.
`presentFrame(which, fb, vp)` = Y7 with `uSrc = frame[prevS|currS]`.

### 3.4 Per-pass uniforms not in the global table

`uBlocks` (ivec2), `uCount` (int), `uRadius` (int), `uStep` (float), `uSrcSize` (ivec2), `uChan` (int), `uMode` (int), `uVpOrigin`/`uVpSize` (vec2), `uFlipY`/`uExact` (int). Values per pass are in the sequence tables above.

### 3.5 The median histogram (changed: a gather)

The extension's version scattered one point per (fine block, component) into a 256×2 histogram with additive blending; the many
equal vectors made the blend serialise. Now `medhist` gathers: each fragment owns 4 bins of one component and one band of 4 parent
rows and sums, over the band's parents, the parent's weight (the number of fine blocks that map to it) where the parent's bin is ≤
the fragment's bin: a cumulative histogram per band, integer weights, exact in fp32. `medsel` binary-searches the cumulative counts
(summed over the bands) for the k1-th and k2-th element. Same median as before. No point draw, no blend, no clear in V1.

### 3.6 Reduce chains (sizes)

Fan-in `uFan` per axis: 16, except the first step of V1's untrusted chain (8: four times the fragments over the full-res trust); step k output size = `ceil(prev/fan)` per axis; at least one step. (changed)

| Chain | Source | Steps (1080p) | Final |
|---|---|---|---|
| `reduce_sad` | `sub2.z`, bw_0×bh_0 = S=2: 120×68 | 120×68 → 8×5 → 1×1 (K=2); S=1: 240×135 → 15×9 → 1×1 (K=2); S=4: 60×34 → 4×3 → 1×1 (K=2) | `sadSum` (Σ block mean SAD; mean = `sadSum/uNumBlocks`) |
| `reduce_untrusted` | `trust` W×H | 1920×1080 → 240×135 → 15×9 → 1×1 (3); 1280×720 → 160×90 → 10×6 → 1×1 (3) | `untrusted` (count of t>0.5) |

---

## 4. Settings

`framegen.js` `settings` object (names = `FH` names; defaults = `FH:68-211`; values only those the core path reads):

| Key | Default | FH line | Shader uniform | Range / note |
|---|---|---|---|---|
| `flowScale` | `2` | FH:211 | `uFlowScale` | **structural**: 1, 2 or 4 (changing it re-allocates all flow-res textures). Quality preset map for the popup: 4 = performance, 2 = quality (default), 1 = full-res (≈2.3× the cost, not better on test clips, FH:207-210) |
| `searchPenalty` | `0.5` | FH:92 | `uLambda` | [0,255] |
| `smoothBonus` | `0.5` | FH:88 | `uSmoothBonus` | [0,255] |
| `hudBonus` | `1.0` | FH:85 | `uZeroBonus` | [0,255] |
| `sceneCutSad` | `30.0` | FH:81 | `uCutSad` | (0,255]; 255 = off |
| `trustLow` | `12.0` | FH:74 | `uTrust0` | [0,255] |
| `trustHigh` | `40.0` | FH:75 | `uTrustScale = fround(1/(trustHigh−trustLow))` | `trustHigh ≥ trustLow+1` |
| `globalFallback` | `0.15` | FH:78 | `uGlobalFrac` | (0,1]; 1 = off |
| `debug` | `false` | – | – | host only: `gl.getError()` after every pass, throw with pass name |
| `motion` | `'search'` | – | – | V1's motion estimate: `'search'` (this contract's block search) or `'learned'` (§10). Ignored by V2. `'learned'` needs `flowScale` 2 (else `RangeError`), `checkSupport(gl).learned` and the net's weights (`__fg.nets`). A change (with V1) drops the estimate like a `motionVersion` switch. (new) |
| `motionNet` | `'e2v5'` | – | – | `'e2v5'` (width 24, e2v4 fine-tuned on more footage: same layout and cost; `__fg.defaultNet`), `'e1v5'` (width 12, e1v4 fine-tuned, about 1.9x faster here; `__fg.smallNet`, the recommended small / fallback net), `'e2v4'` / `'e2v3'` (width 24) or `'e1v4'` / `'e1v3'` (width 12); `motion 'learned'` only; any value needs `webgl/nets/<motionNet>.js` loaded, else `RangeError` naming the file (new; default `'e2v4'` in version '4', `'e2v3'` before) |
| `motionPrecision` | `'fp16'` | – | – | storage of the net's activations: `'fp16'` (half floats, 8 per RGBA32UI texel) or `'fp32'` (RGBA32F, the exact port); the field, pyramid, weights and arithmetic are float32 in both (new) |
| `learnedZeroTest` | `true` | – | – | `motion 'learned'`: the zero-motion test after the net (§10.8); a change drops the estimate (new) |
| `learnedStaticGuard` | `false` | – | – | `motion 'learned'`: the static guard after the synthesis (§10.9); runtime, takes effect at the next `synth` (new) |

Compile-time constants (not settings; defined in shaders / host): BLOCK 8, RADIUS 2, TOP_RADIUS 6, MIN_SIDE 48, sub-pixel steps 0.5 then 0.25 (PY:49), PIX_WIN 5 (5×5 → `1/25`), MISMATCH_BOX 7 (`1/49`), LOOKUP_ITERS 2, 7 per-pixel candidates, max 6 seed candidates, trust threshold 0.5 for the untrusted count. Dropped (game-only / not core): `subpixel` (always true), all `ui*`, `blur*`.

### 4.1 `flowScale` effect

`flowScale` S only changes (a) `fw,fh = rhe(W/S), rhe(H/S)` and therefore level count `L` and every `w_i,h_i,bw_i,bh_i`; (b) `uFlowScale` for `luma` (S×S cell mean) and `fgHalfVec` (flow→full-res upsample ×S, vectors ×S). Full-res passes (`error` … `blit`) are unaffected in cost. Pass counts: §7.

---

## 5. Exactness rules

Correctness is **tolerance-based vs `proto/interp.py`** (D11): mean abs error ≤ 1 level over the output, identical scene-cut / untrusted-fallback decisions. Not bit-exact (no `precise` in ES 3.00, SwiftShader vs RADV).

1. **fp32 everywhere** for intermediates (R32F/RG32F/RGBA32F). **Block SADs are stored as raw sums** over the 64 pixels (consumers multiply by `1.0/64.0`, as the `.comp` do). Luma level 0 is the mean of ≤ S² integer lumas and deeper levels are 2×2 means, so values are dyadic rationals (partial cells with n=3 are the exception): sums are exact in fp32 except at odd level sizes, where the error is ≤ ~1e-5 and only matters at exact ties. Accumulation order is part of the contract: **y outer, x inner** (`seed_sad`, `search_sad`), **rows then sum of rows** (`subpix_sad`), 5×5 window y-outer (`pf_select`).
2. **argmin = strict `<`, ascending candidate index, first wins** (PY `pick()` PY:106-113): seed candidates k=0..5; search candidates row-major `(dy outer, dx inner)`; sub-pixel candidates row-major 3×3; pixfield candidates k=0..6. The cost expression is evaluated exactly as written in the program table (`sum*(1.0/64.0) + penalty`). For `k=0` in `seed_pick` no penalty; for `k≥1` `+uLambda`. For `pf_select` the first candidate always initialises `best`.
3. **No hardware filtering anywhere.** All texture reads are `texelFetch`, all bilinear interpolation is manual in `common.glsl`. Bilinear sites in the reference, and the helper that mirrors each:
   * luma `cv2.remap` (1/32 quantised map): `pf_err`, `subpix_sad` → `fgSampleL`; PY:92 `warp_pair`, PY:180
   * flow field lookup `cv2.remap` (1/32): `lookup` → `fgSampleF`; PY:204
   * RGB warp `cv2.remap` on uint8 (1/32 + rounded to 8 bit): `error` → `fgSampleRGB8`; PY:214
   * flow→full-res `cv2.resize` INTER_LINEAR (no quantisation): `error` → `fgHalfVec`; PY:229
   * block field → flow-res `cv2.resize` INTER_LINEAR: `pf_err`/`pf_select` → `fgPixCand(0,…)`; PY:174
   * INTER_AREA means (arithmetic cell means, not filtering): `luma`, `down`.
   Integer-only sites (`seed_sad`, `search_sad`) use integer vectors ⇒ exact clamped texel taps (`fgFetchL`).
   `LINEAR` filtering is used **only** by `blit` when the viewport size ≠ frame size. `OES_texture_float_linear` is required (D12) but no pass uses it; it is kept as the allowed later optimisation (hardware-bilinear taps in `subpix_sad`/`pf_err`) — not part of this contract.
4. **Clamp-to-edge = `BORDER_REPLICATE`**: every out-of-range read is `clamp(p, 0, size−1)` in the shader (texture wrap is also CLAMP_TO_EDGE, but never relied on). Blocks beyond the level (`bw·8 > w`) read replicated pixels, as PY `pad_to_block`.
5. **Texel-centre conventions:** `texelFetch(t, ivec2(x,y), 0)` addresses texel (x,y) whose centre is (x+½, y+½). `gl_FragCoord.xy` = (x+½, y+½) for texel (x,y) of the bound target (viewport origin (0,0)) ⇒ `ivec2(gl_FragCoord.xy)` is the texel. Positions in `fg*` helpers are cv2-style (integer = pixel centre), so `q − h` with `q = vec2(ivec2)` needs no ±½. The only normalised coordinates in the engine are `blit`'s (and the point positions of `medhist`, which are bin centres).
6. **Rounding of 8-bit things:** luma is rounded to an integer *before* averaging (`fgLuma8`, PY:65); warped RGB is rounded to integer levels (cv2 uint8 remap); e is an integer stored in R8 as `e/255`; the final output is `floor(clamp(o+0.5,0,255))`, matching `np.clip(out+0.5,0,255).astype(uint8)` (PY:246). Mismatch box = mean of the integer e (PY:236 blurs the max-channel float image of those integers).
7. **Thresholds from the reference:** scene cut: `mean_block_SAD > sceneCutSad` (30) where the mean is Σ over all `bw_0·bh_0` padded blocks of the *final* (¼ px) winning block mean-SAD (PY:164,190,225; FG fixed-point sum replaced by an fp32 reduce). Untrusted: pixel untrusted iff `t > 0.5`; global fallback iff `count > globalFallback · W·H` (PY:238). Per-pixel trust ramp `clamp((e − trustLow)/(trustHigh − trustLow), 0, 1)`. Cut is evaluated first; either decision outputs the **nearer real frame** (`prev` for t ≤ 0.5, `curr` for t > 0.5).
8. **Level / cell sizes use round-half-even** (`rhe`), mirroring `cv2.resize(fx=…)`'s `saturate_cast<int>`; the *.comp* average only the valid pixels of partial cells. `cv2.INTER_AREA` equals this cell mean exactly only if `W,H` are divisible by S and every level size is even; harness pairs should use dimensions divisible by 4·S and whose flow-res level sizes stay even down the pyramid (e.g. 640×384 at S=2: 320×192, 160×96, 80×48) to keep the reference comparison tight. Odd sizes follow the `.comp` behaviour, not cv2's fractional-area weights.
9. **Frame content:** engine consumes RGB from RGBA8 (alpha ignored). `interp.py` operates on BGR — the harness must convert BGR→RGBA before upload (luma weights are R 4899 / G 9617 / B 1868 of 16384).

### 5.1 y-orientation (single convention)

* **Every engine texture stores image row 0 (top) first in memory/texel row 0.** `pushFrame` uploads with `UNPACK_FLIP_Y_WEBGL = false`, so a `HTMLVideoElement`/`ImageBitmap`/canvas lands with its top row at texel row 0; a raw `{data}` upload is `W` RGBA pixels per row, **rows top-first** (numpy `(H,W,4)` C-order layout, harness uploads it as-is).
* All passes work in texel coordinates (row 0 = top), so nothing is flipped internally and `y` in this document always means "texel row".
* **`readPixels` on a framebuffer whose attachment is a texture returns texel row 0 first** (window y = texel y), i.e. top-first like numpy. `debugRead` and the harness' own FBO therefore need **no flip**.
* **Only the default framebuffer (canvas) is bottom-up.** `blit` compensates (`uFlipY=1` exactly when `framebuffer === null`), so the canvas shows the image upright. Anyone calling `readPixels` on the *canvas* gets bottom-first rows — the golden harness must not do that; it passes its own FBO to `synth` (or reads `debugRead('out')`).
* A caller-supplied FBO (texture-backed) is treated as a texture: no flip. `synth`/`presentFrame` accept an options bag `{flipY}` to override the default `framebuffer === null`.

### 5.2 Other numeric notes

* `final` computes `meanSad = sadSum/uNumBlocks`; the reference accumulates `uint(roundEven(sad·256))` per block (1/256 fixed point). The fp32 sum differs by ≤ ~1e-3 in the mean; only matters for a pair within that of the threshold.
* Constants folded in double then `fround`ed on the CPU (`uS`, `uWA`, `uWC`, `uTrustScale`) so that the float32 values equal numpy's `np.float32(...)` constants (PY:198,213,243).

---

## 6. `framegen.js` API (host)

Classic script. Attaches **`globalThis.__fg`**; defines nothing else globally. No `fetch`, no DOM access except what is passed in.

```js
__fg.version                       // string, '5' (changed; history below, README "API" lists the additions)
__fg.requiredExtensions            // ['EXT_color_buffer_float','EXT_float_blend','OES_texture_float_linear']
__fg.checkSupport(gl, frame?)      // -> { ok, missing: string[], sizeError: string|null, quality: boolean, qualityError: string|null, learned: boolean, learnedError: string|null } (changed)
__fg.nets                          // { e2v5, e1v5, e2v4, e1v4, e2v3, e1v3 }: the learned estimator's weights, attached by webgl/nets/<name>.js (§10; new)
__fg.defaultNet                    // 'e2v5' (read-only; = defaultSettings().motionNet; new)
__fg.smallNet                      // 'e1v5' (read-only): the recommended small / fallback net; set motionNet: __fg.smallNet to drop to it (new)
__fg.shaderFiles                   // string[] — every key `sources` must contain (§1), e.g. 'common.glsl', 'vert.glsl', 'estimate/luma.frag.glsl', ...
__fg.defaultSettings()             // -> fresh copy of the defaults of §4
__fg.levelLayout(W, H, flowScale)  // pure -> { fw, fh, levels: [{ w, h, bw, bh, radius }] }  (§2 sizes)
__fg.createEngine(gl, sources, settings) // -> engine ; throws Error('framegen: <program>: <info log>') on compile/link failure
```

* `checkSupport(gl, frame?)` (changed): `missing` contains `'webgl2'` if `gl` is not a `WebGL2RenderingContext` (then nothing else is probed), otherwise each name of `requiredExtensions` for which `gl.getExtension(name)` is null, plus `'MAX_DRAW_BUFFERS>=3'` / `'MAX_VERTEX_TEXTURE_IMAGE_UNITS>=1'` if those limits fail; feature names only. `sizeError` = `frameSizeError(gl, frame.width, frame.height, frame)` for the optional `frame = { width, height, flowScale }`, else null (before, that text was pushed into `missing`). `ok = missing.length === 0 && !sizeError`. `quality` / `qualityError`: V2 Quality's MIN-blend probe (README "API": a 1 × 1 R32F framebuffer, three points, read back; cached per context; restores the GL state it touches); `quality` does not affect `ok`, and with anything in `missing` it is false with `qualityError = 'needs ' + missing`. It calls `getExtension` (so it also *enables* the extensions). The probe allocates (and deletes) one 1 × 1 texture, framebuffer, program and VAO, the first time per context.
* `version` history: `'5'`: nets e2v5 (now the default `motionNet` and `defaultNet`) and e1v5 (now `smallNet`), ml-fg's fine-tunes of e2v4 / e1v4 on more varied footage (same architecture, layout and cost); bumped because the default learned output and `smallNet` changed, the API is unchanged; e2v4 / e1v4 / e2v3 / e1v3 stay selectable. `'4'`: nets e2v4 (now the default `motionNet`) and e1v4, `learnedZeroTest` (default `true`: the learned default output changed), `learnedStaticGuard`, `defaultNet` / `smallNet`. `'3'`: the learned estimator (`motion`, `motionNet` default `'e2v3'`, `motionPrecision`, `__fg.nets`, `checkSupport` `learned`, `stats()` `duplicate`). `'2'`: V2 Quality, RGBA16F input, `synthBlur`, `frameSizeError`, `checkSupport` `sizeError` / `quality`. Nothing was removed in any of them.
* `defaultNet` / `smallNet`: non-writable properties of `__fg` (assigning to them is ignored, or throws in strict mode); a host that switches to a cheaper net reads `smallNet` instead of naming one, so a newer small net needs no host change. Both nets' scripts must be loaded to select them.
* `createEngine` / `setSettings` (changed): where `checkSupport(gl).quality` is false, settings with `motionVersion: 2, v2Mode: 'quality'` throw a `RangeError` naming `qualityError`; the engine keeps its settings. `engine.support` is the `checkSupport(gl)` result of `createEngine`.
* `createEngine(gl, sources, settings)`: `gl` is a `WebGL2RenderingContext`; `sources` maps path-relative-to-`shaders/` → string (`{ 'common.glsl': str, 'vert.glsl': str, 'estimate/luma.frag.glsl': str, … }`); `settings` is partial (missing keys take §4 defaults). Calls `checkSupport` and throws if not ok. Compiles and links every program (assembly per §1.1) and creates the empty VAO; textures and FBOs are allocated lazily on the first `resize`/`pushFrame`.

**engine**

```js
engine.settings                        // frozen current settings
engine.info                            // { W, H, fw, fh, flowScale, levels:[{w,h,bw,bh,radius}] } | null before first resize
engine.setSettings(partial)            // -> { rebuilt: boolean }; runtime keys apply to the next call; changing flowScale re-allocates (drops frame history)
engine.resize(w, h)                    // (re)allocates all W×H / flow-res textures; discards pushed frames and the estimate
engine.pushFrame(source)               // -> number (count of frames pushed since resize); upload + luma + pyramid (§3.3)
engine.estimate()                      // -> boolean; false (no-op) if fewer than 2 frames pushed
engine.synth(t, framebuffer, viewport, opts)  // -> boolean; false if no estimate. t in (0,1)
engine.presentFrame(which, framebuffer, viewport, opts) // which: 'prev' | 'curr'; blit of a real frame
engine.stats()                         // debug readback, sync stall (below)
engine.debugRead(name)                 // debug readback (below)
engine.passCounts()                    // -> { pushFrame, estimate, synth, clears } draw counts of the last call of each (for the §7 table)
engine.dispose()                       // deletes every texture/FBO/program/VAO it created; engine unusable after
```

* **`pushFrame(source)`** — `source`: `HTMLVideoElement` (size `videoWidth×videoHeight`), `ImageBitmap`, `HTMLCanvasElement`/`OffscreenCanvas`/`ImageData`-like (size `width×height`), **or** `{ width, height, data: Uint8Array|Uint8ClampedArray }` raw RGBA, `data.length === 4*width*height`, rows top-first. If the size differs from `engine.info` (or no `resize` yet) it calls `resize(width,height)` first (history dropped). Sets `UNPACK_FLIP_Y_WEBGL=false`, `UNPACK_PREMULTIPLY_ALPHA_WEBGL=false`, `UNPACK_ALIGNMENT=1` (raw), leaves `UNPACK_COLORSPACE_CONVERSION_WEBGL` at its default; uploads with `texSubImage2D(TEXTURE_2D,0,0,0,w,h,RGBA,UNSIGNED_BYTE,…)` (raw) or the 7-argument `texSubImage2D(TEXTURE_2D,0,0,0,RGBA,UNSIGNED_BYTE,source)` form (everything else). Throws if the browser rejects the source (e.g. tainted video: `SecurityError` propagates).
* **`synth(t, framebuffer, viewport)`**: `framebuffer` is a `WebGLFramebuffer` (draw target, texture-backed, treated as top-first), `null` (default framebuffer / canvas, flipped), or `undefined` (render `out` only, no blit; for `debugRead('out')`). `viewport` = `{ x, y, width, height }` in target pixels, default `{0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight}` for `null` and `{0,0,W,H}` otherwise. `t == 0.5` takes the exact 2× path (no `lookup`, `uMid=1`). Valid only until the next `pushFrame`. Leaves `framebuffer` bound, `gl.viewport` = `viewport`.
* **GL state:** the engine owns the context while a call runs: it disables `DITHER, BLEND` (except inside V2 Quality's splat and the blur accumulation), `DEPTH_TEST, STENCIL_TEST, SCISSOR_TEST, CULL_FACE`, sets `colorMask(true,true,true,true)`, binds its own VAO/program/FBO/textures (clobbering texture units 0..6 and `ACTIVE_TEXTURE`), and restores nothing. The runtime must use a context that nothing else draws with (D17).
* **`stats()`** → `{ meanSad: number, cut: boolean, untrusted: number|null, untrustedShare: number|null, flick: boolean|null, fallback: null|'cut'|'untrusted', t: number|null, duplicate: boolean|null, dupDiff: number|null }` (`duplicate`, `dupDiff`: §10, null unless the estimate was the learned one). `meanSad = sadSum/(bw_0·bh_0)`; `cut = meanSad > sceneCutSad`; `untrusted`/`untrustedShare = untrusted/(W·H)`/`flick = untrusted > globalFallback·W·H` come from the **last `synth` call** (null if none yet); `fallback` mirrors `interp.synth(..., why=True)`: `'cut'` if cut, else `'untrusted'` if flick, else null. Uses `readPixels` on the 1×1 textures — debug/harness only, never on the hot path.
* **`debugRead(name)`** → `{ name, width, height, channels, data }` where `data` is row-major, **top row first**, channels interleaved (`index = (y*width + x)*channels + c`): `Float32Array` for float formats, `Uint8Array` for RGBA8/R8, `Uint32Array` for R16UI. `channels` = channels of the internal format (R→1, RG→2, RGBA→4). Valid names (from §2): `frame.prev`, `frame.curr`, `luma.prev.<i>`, `luma.curr.<i>`, `hist.<i>` (changed), `fieldRaw.<i>` (new), `median.<i>`, `seedSad.<i>`, `seed.<i>`, `searchSad.<i>`, `field.<i>`, `subSad`, `sub1`, `sub2`, `sadSum`, `pfErr0`, `pfErr1`, `pixfield`, `fieldT`, `warpA`, `warpC`, `mismatch`, `hsum`, `trust`, `untrusted`, `out`, and the learned estimator's `ml.*` (§10). (`prev`/`curr` are resolved through the ping-pong at call time.) Unknown or not-yet-written name → throws. Implementation: temporary FBO on the texture, `readPixels` with `RGBA/FLOAT` (float), `RGBA/UNSIGNED_BYTE` (8-bit), `RGBA_INTEGER/UNSIGNED_INT` (R16UI), then channel-strip; no flip.

### 6.1 `debugRead` ↔ `interp.py` map (for the harness)

| `debugRead` | `interp.py` | Notes |
|---|---|---|
| `luma.prev.0`, `luma.curr.0` | `ya`, `yc` from `flow(a,c)` (PY:150-151, returned PY:164) | R32F, fw×fh; compare ≤1e-3 |
| `luma.*.<i>` | `pyramid(ya)[i]` (PY:73) | level order matches (0 = finest) |
| `field.<i>` | per-level `v` after `median3` in the `flow()` loop (PY:157-161): level index `i` ↔ `reversed(pa)` position; **integer-valued for i ≥ 1; for i = 0 it is the integer pre-subpixel field** | harness re-runs the PY loop to capture intermediates |
| `sub2.xy` | `v` returned by `flow()` (PY:164) | block half-vectors, bh×bw×2 |
| `sub2.z`, `sadSum/(bw_0·bh_0)` | `sad` per block / `float(sad.mean())` (PY:164) | decision: `> CUT_SAD` |
| `pixfield` | `pixel_field(v, ya, yc)` (PY:167) | fh×fw×2 |
| `fieldT` | `field_at(pf, t)` (PY:194) | t≠0.5 |
| `warpA`, `warpC` | `warp_t(...)` outputs `wa`,`wc` (PY:232) | uint8 round |
| `mismatch` | `np.abs(wa-wc).max(2)` (pre-blur), integers | |
| `trust` | `np.clip((e-TRUST[0])/(TRUST[1]-TRUST[0]),0,1)` (PY:237) | |
| `stats().untrustedShare`, `.fallback` | `(u>0.5).mean()`, `synth(..., why=True)[1]` (PY:238) | decisions identical |
| `out` | `interpolate(a,c,t)` (PY:249) | RGBA vs BGR channel order handled by the harness |

---

## 7. Pass-count / cost estimate

Draw calls (a clear = `clearBufferfv` on `hist`, counted separately). 1080p, `K_sad = 2`, `K_untr = 3` (see §3.6).

* `pushFrame` = `1 + (L−1) = L` draws.
* `estimate` = `3` (top level: search_sad, search_pick, median3) `+ 7·(L−1)` (medhist, medsel, seed_sad, seed_pick, search_sad, search_pick, median3 per lower level) `+ 4` (sub-pixel) `+ K_sad` `+ 2` (pixfield) draws, no clears. (changed: the table below is the extension's original count, +L draws −(L−1) clears now)
* `synth(t)` = `8` draws at t = 0.5 (error, hbox, vbox, 3× reduce, final, blit), `9` otherwise (+lookup). `presentFrame` = `1`.

| flowScale | L | pushFrame | estimate | synth t=.5 / t≠.5 | **2×** per source frame (push + est + synth(.5) + present) | **3×** (push + est + 2·synth(≠.5) + present) | **4×** (push + est + synth(.25) + synth(.5) + synth(.75) + present) |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 5 | 34 | 8 / 9 | 5+34+8+1 = **48** | 5+34+18+1 = **58** | 5+34+26+1 = **66** |
| **2 (default)** | 4 | 4 | 28 | 8 / 9 | 4+28+8+1 = **41** | 4+28+18+1 = **51** | 4+28+26+1 = **59** |
| 4 | 3 | 3 | 22 | 8 / 9 | 3+22+8+1 = **34** | 3+22+18+1 = **44** | 3+22+26+1 = **52** |

The scout's "~35 / ~51" (2× / 4×) counted fewer passes (no blit/present, fused trust); this design has 41 / 59 at the default preset. Extra hist clears: 3 at S=2.

Texture-fetch order of magnitude per pair at 1080p, S=2 (dominant terms): `search_sad` level 0 `25·8160·128 ≈ 26 M` + lower levels ≈ 10 M; `seed_sad` ≈ 6 M; `subpix_sad` `2·9·8160·64·8 ≈ 75 M` (heaviest; the first optimisation target); `pf_err` `518 k·7·8 ≈ 29 M`; `pf_select` ≈ 26 M vec4 fetches; ⇒ ~2·10⁸ fetches, in line with the scout's 0.3–0.6 ms/pair on the 7900 XTX. Per synth: `error` ≈ 2 M·(4+8) ≈ 25 M, plus ~100 MB of full-res read/write across `hbox/vbox/final/blit`. Per-GL-call CPU cost on a presented canvas is unmeasured (scout risk): ≈ 100+ GL state changes per source frame at 4×.

---

## 8. Departures from the scout's mapping (for implementers)

1. **Candidates are tiled, not strip-laid:** `searchSad`/`subSad` use a `D×D` / `3×3` tile per block so the argmin reads each block's candidates in reference row-major order from one contiguous tile (§2).
2. **`median3`**: the extension fused it into `search_pick` by recomputing the argmin of the 9 neighbour blocks (9× the work: the most expensive pass); here the raw argmin is stored (`fieldRaw[i]`) and `median3` is a 9-fetch pass. (changed)
3. **`seed_pick`** is its own tiny pass (the scout's "seed cost + argmin"); the global-median `medsel` is separate from the `GL_POINTS` draw.
4. **Stats are two separate R32F 1×1 textures** (`sadSum` per pair, `untrusted` per synth) read directly by `final`, not one RG32F stats texture (the two come from different phases/owners). No flag texture: the `t > 0.5` test happens in the first untrusted-reduce step.
5. **`error` writes `warpA`/`warpC`** (MRT) so `final` does not recompute the warps (the reference's `resolve.comp` calls `mismatch()` a second time).
6. **`blit` pass + `presentFrame`** are added for presentation: scaling to the canvas, the default-framebuffer y flip, and showing real frames through the same path (D9).
7. **`MAX_DRAW_BUFFERS` 8 is not needed** (max 3; the learned estimator's match pass 4, §10); `OES_texture_float_linear` is required by D12 but unused (§5 rule 3).
8. **"Integer-valued SAD"** holds exactly only for S=1 level 0 (luma is an integer there); elsewhere luma is dyadic, and the fp32 sums are still exact except at partial cells (§5 rule 1).

---

## 9. Motion version 2

See README.md (V2 Performance and Quality, the static mask, frame formats, motion blur, tests).

---

## 10. The learned motion estimator (`motion: 'learned'`) (new)

An opt-in replacement of V1's motion estimate (§3.3 E1–E14) by ml-fg's learned estimator; V1's synthesis (§3.3 Y1–Y7) is
unchanged and consumes its output. The porting contract is ml-fg's `export/SPEC.md` (stages 1–7, abbreviated **SP**) with its
executable reference `export/reference.py` (**REF**); the nets are `e2v5` (width 24, 95,530 weights, the default: e2v4 fine-tuned, identical
architecture and weight layout), `e1v5` (width 12, 28,330 weights, the small net `__fg.smallNet`: e1v4 fine-tuned, identical
layout), `e2v4` (width 24, e2v3 fine-tuned), `e1v4` (width 12, e1v3 fine-tuned), `e2v3` (width 24, 95,530 weights) and `e1v3` (width 12, 28,330 weights). Cites are SP sections.

### 10.1 Contract with V1

The estimate writes exactly what V1's search leaves for the synthesis, nothing else:

* `pixfield` (RG32F, fw × fh): the half-vector field `h`, flow-res pixels, texel (x, y) = (hx, hy), row 0 = top (SP 0, 5.3).
* `sadSum` = the last texel of the `sadRed` chain (1 × 1 R32F) = `sad · bw₀ · bh₀`, so that `meanSad = sadSum / (bw₀ · bh₀)`
  (`final`, `stats()`) is the net's `sad` (SP 6: the mean over the fw × fh flow pixels of `|ya(q − h) − yc(q + h)|`, true
  bilinear, clamp to the unpadded grid, 0..255). The cut rule `meanSad > sceneCutSad` (30) is unchanged.

V1's own estimate textures (`field.*`, `sub2`, `pfErr*`, …) are not written in learned mode (they keep stale data). `flowScale`
must be 2 (SP 5.5). RGBA16F input works through the frame-format helpers (luma = the cv2 weights below on `FG_LV` levels,
unrounded; the duplicate test on `FG_D`).

### 10.2 Duplicate pairs

The net has no zero-motion prior: on a == c it outputs motion (SP 9 (a)). The estimate therefore applies FGR's duplicate test —
the mean of `|a − c|` over every pixel and the R, G, B channels at most **0.6** levels (`proto/harness.py:29,50` `DUP_DIFF`,
`gpu/compare.py:56,63`) — on the GPU: `lv0` writes per flow texel the sum over its pixels (`ml.dupSrc`), `reduce_sad` sums it to
1 × 1 (`ml.dup`), and every later pass reads that texel (`mlDup()` in `ml_common.inc.glsl`) and, on a duplicate, writes zeros
without doing its work: `pixfield` = 0, `sadSum` = 0, so the synthesis shows the input unchanged. No readback, no CPU branch.
`stats()` reports `duplicate` and `dupDiff` (the mean) after a learned estimate.

### 10.3 Textures (allocated by the first learned estimate; again on a size, net or precision change)

Padded flow grid `PW × PH` = fw, fh rounded up to a multiple of 16 (SP 3); level l is `(PW >> l) × (PH >> l)`, l = 0..4, each an
exact half of its parent.

| Name (`debugRead`) | Format | Size | Content |
|---|---|---|---|
| `ml.pyr.0` | RGBA32F | PW × PH | la, lc (= ya / 255, yc / 255, level 0 of the net), ya, yc (0..255); edge replicated beyond fw × fh |
| `ml.pyr.<l>`, l = 1..4 | RG32F | level | la, lc: `((p00 + p01) + (p10 + p11)) · 0.25` |
| `ml.dupSrc` | R32F | PW × PH | the duplicate test's per-texel sums (0 in the padding) |
| `ml.x.<k>` | fp32: RGBA32F × 3 + RG32F; fp16: RGBA32UI × 2 | PW × PH | the 14-channel input of the level being computed (SP 4b order) |
| `ml.act.<s>.<k>` | fp32: RGBA32F × G; fp16: RGBA32UI × ⌈G/2⌉ | PW × PH | ping-pong activations, set s = 0, 1 (G = width / 4) |
| `ml.h.<l>`, l = 1..4 | RG32F | level | the field after level l (REF `dbg['h'][l]`) |
| `ml.dup`, `ml.sad` | R32F | 1 × 1 | the sums of `|a − c|` and of the SAD terms (chains `dupRed` fan 4/4/…, `sadRed` fan 2/4/…) |
| `ml.pfNet` | RG32F | fw × fh | zero test on: the net's field (level 0's last conv writes it instead of `pixfield`) |
| `ml.ztCost` | RG32F | fw × fh | zero test: r = d = `|ya(q − h) − yc(q + h)|`, g = e = `|ya − yc|` |
| `ml.zt` | RGBA32F | fw × fh | zero test: r = the SAD term `zero ? e : d`, g = C(h), b = C(0), a = zero (1 / 0) |
| `sg` | RG16F | W × H | static guard: r = d (3 × 3 mean of s), g = dMin (9 × 9 min); `sgH` (RG16F, its horizontal half) is not readable |

`x` and `act` are level-0 sized; a coarser level renders into their top-left corner (viewport = the level) and the convs bound
their taps by the level size `uSize` (zero padding at the padded level's border, SP 4c pitfall), so data of other levels beyond
it is never read. fp16 packs 8 channels per texel: texel component k = `packHalf2x16(channel 2k, channel 2k + 1)` of the
texture's 8. Memory at 1080p (PW × PH = 960 × 544): about 77 MB (e2v5 / e2v4 / e2v3, fp16), 61 MB (e1v5 / e1v4 / e1v3, fp16), 140 MB (e2v5 / e2v4 / e2v3, fp32), 90 MB
(e1v5 / e1v4 / e1v3, fp32); 1440p ×1.76.

### 10.4 Programs and sequence

| Program | File (defines) | Inputs (units) | Output |
|---|---|---|---|
| `ml_lv0` | `lv0` | uPrev, uCurr | MRT `ml.pyr.0` + `ml.dupSrc`: SP 1–3 luma (`(3735 B + 19235 G + 9798 R + 16384) >> 15`, cv2 5.0.0), 2 × 2 area mean, edge pad, / 255 |
| `ml_down` | `down` | uSrc | `ml.pyr.<l>` |
| `ml_match`, `ml_match_h` | `match` (`ML_PACKED` for fp16) | uPyr, uHc, uDup | MRT `ml.x.*`: SP 4a (`mlFieldUp`: y then x, ×2) + 4b (18 true-bilinear taps) |
| `ml_conv<CIN>x3`, `ml_last<G>` | `conv` (`ML_CIN`, `ML_COUT`, `ML_LAST`) | uIn0..uIn<CIN−1>, (uHc), uDup | fp32: 3 RGBA32F per pass (width 24: two passes per layer), or the field |
| `ml_h<CIN>g<CG>o<COUT>[L]` | `conv_h` (`ML_CIN`, `ML_CG`, `ML_COUT`, `ML_COUTA`, `ML_LAST`) | same, usampler2D | fp16: ⌈COUT/2⌉ RGBA32UI per pass (one pass per layer), or the field |
| `ml_sad` | `sad` | uLv0, uField | first step of `sadRed` |
| `ml_sadout` | `sadout` | uSrc, uDup | `sadRed[last]` of V1 = sum · bw₀ · bh₀ / (fw · fh) (`uWeight`) |
| `ml_zt_cost` | `zt_cost` | uLv0, uField (`ml.pfNet`), uDup | `ml.ztCost` |
| `ml_zt` | `zt` | uCost, uField (`ml.pfNet`), uDup | MRT `pixfield` + `ml.zt` |
| `ml_sg_h`, `ml_sg_v` | `sg_h`, `sg_v` (include `sguard.inc`) | uPrev, uCurr / uH | `sgH` / `sg` |
| `final_sg` | `synth/final` (`FG_SGUARD`, include `sguard.inc`) | final's 7 + uSg (unit 7), uField (8) | `out` |

`estimate()` in learned mode: `ml_lv0`, `reduce_sad` × |dupRed|, `ml_down` × 4, then for l = 4..0: `ml_match.l`, `ml_c0.l` …
`ml_c4.l` (fp32, width 24: `ml_c<k>a.l`, `ml_c<k>b.l` for k < 4), the last one writing `ml.h.l` (l > 0) or `pixfield` (l = 0, over
fw × fh; its taps read the padded level), then `ml_sad`, `reduce_sad` × (|sadRed| − 1), `ml_sadout`. Draws at 1080p: 47 (fp16,
every net; fp32 width 12), 67 (fp32, width 24). With the zero test (the default) the last conv of level 0 writes `ml.pfNet`, and
`ml_sad` is replaced by `ml_zt_cost`, `ml_zt`, `reduce_sad` × |sadRed| (two draws more at 1080p: 49 / 69). Synth with the static guard: `ml_sg_h`, `ml_sg_v` (the first guarded synth of a pair) and `final_sg` for `final`. The pass names (with the level) are what `setPassHook` sees.

Conv (SP 4c, REF `conv3x3`): `out[o](q) = b[o] + Σ_tap Σ_i W[o][i][ky][kx] · in[i](q + (kx − 1, ky − 1))`, hidden layers
`clamp(·, 0, 1)`, the last layer `h + 2 · out` with h recomputed in float32 by `mlFieldUp` (it is never stored in the
activations). Arithmetic float32 throughout.

### 10.5 Weights

`webgl/nets/<name>.js` (generated by `tests/make_nets.py` from ml-fg's `export/<name>/manifest.json` + `weights.bin`, sha256
checked) attaches `__fg.nets[name] = { name, width, levels, depth, inChannels, padMultiple, flowScale, cutSad, levelToStack,
stacks: [[{ w, b, cin, cout, act }]], count, sha256, checkpointSha256, checkpointStep, weights }`, `weights` = base64 of the
float32 little-endian stream (OIHW, SP 3), `w` / `b` element offsets. The engine decodes it once (`atob`) and uploads one
uniform buffer per net (e2v* about 400 KB, e1v* about 122 KB): per level, layer and 12 output channels a std140 block
`{ mat4 uW[COUT · 9 · CG]; vec4 uB[COUT]; }` at a multiple of `UNIFORM_BUFFER_OFFSET_ALIGNMENT`, `uW[(o · 9 + tap) · CG + g]`
= the 4 × 4 weights from input channels 4g..4g+3 (columns) to output channels 4o..4o+3 (rows), zero beyond the layer's channels
(the 14-channel input is 16 slots). A pass binds its block(s) with `bindBufferRange` to binding points 0 (`MlW`) and 1 (`MlW2`:
fp16, outputs 12..23 of a 24-wide layer). The host checks each program's `UNIFORM_BLOCK_DATA_SIZE` against its packing.

### 10.6 Required features

WebGL2 core minimums only (`checkSupport().learned` / `learnedError`): `MAX_DRAW_BUFFERS` and `MAX_COLOR_ATTACHMENTS` ≥ 4 (the
match pass), `MAX_UNIFORM_BLOCK_SIZE` ≥ 10,416 bytes, `MAX_FRAGMENT_UNIFORM_BLOCKS` ≥ 2, `MAX_TEXTURE_IMAGE_UNITS` ≥ 9 (the static guard's `final_sg`), plus
the backend's `EXT_color_buffer_float` (RGBA32F / RG32F / R32F targets; RGBA32UI is core). Missing → `createEngine` /
`setSettings` throw a `RangeError` for learned settings and keep the old ones; so does a net whose `webgl/nets/<name>.js` was
not loaded.

### 10.7 Exactness

`motionPrecision 'fp32'` is an exact float32 port: ya / yc bit-exact, the coarsest level within 2.5e-5 of the torch vectors,
pf mean ≤ 5e-5 px / max ≤ 1.8e-2 px, sad relative ≤ 2e-6, identical cut on every test pair of both nets (README "Learned
motion estimate"). `'fp16'` rounds the stored input and activations to half floats (SP 7, its expected first optimisation): pf
differs pointwise (low-texture pixels), the synthesised frame stays ≥ 63 dB from the reference's and its PSNR to the real middle
frame within 0.002 dB of the reference's. Differences from REF: the GPU's division / FMA contraction (noise amplified by the
net, SP 7), nothing systematic; hardware bilinear is not used (4 `texelFetch`).

### 10.8 The zero-motion test (`learnedZeroTest`, default on)

SP stage 7 / REF `zero_test()`, between the net (SP 5) and the SAD (SP 6). Inputs: the net's field `ml.pfNet` (fw × fh, the
net's level 0 writes it instead of `pixfield`; the 5 × 5 windows read the neighbours' net vectors, so the field cannot be
replaced in place) and `ya`, `yc` = `ml.pyr.0` b, a (the unpadded fw × fh planes the net got, 0..255).

* `ml_zt_cost`, per flow pixel q: d = `|mlBil(ya, q − h) − mlBil(yc, q + h)|` (true bilinear, clamp to fw × fh, h = the net's
  vector at q), e = `|ya(q) − yc(q)|` → `ml.ztCost`.
* `ml_zt`: C(h), C(0) = the 5 × 5 means of d, e with replicate border (each tap's coordinate clamped to the frame; REF `box`:
  the 5 taps of a row left to right, the 5 row sums top to bottom, / 25), `zero = (C(0) − 1.0) <= C(h)`; MRT att0 `pixfield` =
  zero ? 0 : h, att1 `ml.zt` = (zero ? e : d, C(h), C(0), zero).
* `reduce_sad` sums `ml.zt`.r over the fw × fh grid (the `sadRed` chain, fan 2 then 4), `ml_sadout` scales it as before: by SP 7's
  identity (a zeroed pixel's SAD term is e, a kept one's d) this is SP 6's `sad` of the final field, so `sadSum`, `meanSad` and the
  cut belong to the field after the test.

Constants `ML_ZERO_WIN` 5 and `ML_ZERO_BONUS` 1.0 (`zt.frag.glsl`) are SP's fixed values. Everything is float32 (SP 7: fp16 cost
maps were not tested), in both `motionPrecision`s. Duplicates: both passes write 0 (the net field is 0 then anyway). Off: the last
conv writes `pixfield` and `ml_sad` runs, exactly as without the test. Exactness (`tests/learned.mjs` against the vectors'
`*_zt` files, whole chain = the engine's own net field through its own test): fp32 mask agreement 99.998–100 % (0–5 flipped
pixels per pair; REF itself flips 0–6), pf on the agreeing pixels within SP 8's pf tolerances, `sad_zt` relative ≤ 8e-5,
identical cut, C(0) within 1.5e-5 levels; fp16 mask agreement 99.88–99.997 %.

### 10.9 The static guard (`learnedStaticGuard`, default off)

ml-fg `scripts/static_guard.py` (authoritative; `superdoc/features/static-guard.md`), a synthesis-stage pass with the learned
estimate only. Constants: the marked block at the top of `learned/sguard.inc.glsl` (`SG_LO` 7.65, `SG_HI` 22.95, `SG_HALO_LO`
5.1, `SG_HALO_HI` 7.65, `SG_HALO` 4, `SG_MOTION_LO` 0.5, `SG_MOTION_HI` 1.0, `SG_MIN_W` 0.001 = static_guard.py `DEFAULTS`).

* s(p) = max over R, G, B of `FG_D(prev, curr)` (§1.2 frame formats: the levels of the mismatch metric; Rgba8 integer levels,
  RgbaF16 value × 255 or, for linear input, the PQ code × 255).
* `ml_sg_h` (full res, RG16F `sgH`): r = Σ s over x − 1..x + 1, g = min s over x − 4..x + 4 (replicate border); `ml_sg_v` (RG16F
  `sg`): r = d = Σ of 3 rows / 9 (cv2.blur), g = dMin = min of 9 rows (cv2.erode). They depend on the frames only and run once
  per pair, at its first guarded synth (re-armed by `pushFrame`).
* `final_sg` = `synth/final.frag.glsl` with `FG_SGUARD`: after the fallbacks (cut, flick: returned unchanged, as
  `static_guard.wrap`) and V1's blend `o` (Rgba8: rounded to the stored 8-bit value first), motion = 2 max(uWA, uWC) ·
  flowScale · |field|, the field magnitude (the field at t: `pixfield` at t = 0.5, else `fieldT`) bilinearly upsampled with
  cv2 `INTER_LINEAR`'s mapping (half-pixel centres, scale fw / W, clamp); gate = 1 − ss(0.5, 1.0, motion); wLocal = (1 −
  ss(7.65, 22.95, d)) · (1 − ss(5.1, 7.65, d) · (1 − gate)); wHalo = 1 − ss(5.1, 7.65, dMin); w = max(wLocal, wHalo) · gate,
  ss = static_guard.py's smoothstep. Where w > 0.001: o = o (1 − w) + (uWA · prev + uWC · curr) w (working values), then
  `fgStore` (Rgba8: round half up, as static_guard.py's `+ 0.5` and truncation).

Off: `final` as before (byte-identical). Exactness (`tests/learned.mjs` vs static_guard.py on the engine's own a, c, unguarded
output and field at t): max |d| 0 (SwiftShader) or 1 level on 1–4 px of 230,400 (hardware GL: rounding ties) for RGBA8, ≤ 0.33 levels for RGBA16F.
