// V2's synthesis, included by v2_morph.comp (Performance) and v2_qmorph.comp (Quality, V2_QUALITY: the additions are described in
// v2_quality.glsl), and with V2_STATIC by v2_morph_s.comp / v2_qmorph_s.comp. One full-resolution pass (recordSynth: n = 1;
// recordSynthBlur: n sub-frames averaged in registers, each exactly what recordSynth(t_k) gives), then a static pass that redoes the
// cells near a static overlay, then a fix-up pass that is a no-op unless the whole-frame flick fallback fired:
//   g   = the cleaned block field upsampled bilinearly to this pixel (v2_motion.glsl hvAt; Quality: its per-pixel field, hvQ), taken
//         to time t by the fixed point g = h(q - (2t - 1) g) (2 steps from h(q): lookup.comp's, here per pixel on the smooth field);
//   out = (1 - t) prev(q - 2t g) + t curr(q + 2(1 - t) g), both taps bilinear, CrosshairV2's tap rejection on the pair (Quality:
//         the weights lean to the frame that still shows the content, a tap on a static overlay takes the other, a blur along the
//         motion where the taps disagree);
//   blended toward the zero vector's (1 - t) prev(q) + t curr(q) where the two frames show q (and a line of its neighbours) the same
//   but the morph's taps disagree: HUD text and other static overlays, static background that the smooth field would drag along;
//   a tap outside the frame takes the other tap; where the taps disagree badly, one of the 4 block vectors around the content's
//   place may take over as a rigid motion of the pixel's 4x4 group (no blend of two very different vectors: ring outlines).
// Static overlays (both modes, V2's static-structure mask S and its enclosure E, v2_static.comp / v2_enclose.comp; where both real
// frames show the same structure, also a semi-transparent HUD panel whose background moves): a pixel on S (confirmed at full
// resolution) or inside E whose taps leave it takes the zero vector, and a tap that lands on S / E while the field moves takes the
// other tap. Most of a frame is far from every overlay, so the main pass (no V2_STATIC) has none of it; the static pass (fix = 2,
// V2_STATIC) takes the 8x8 tiles whose pixels or taps come near one (v2_enclose.comp's C) and leaves the others at once; it runs
// beside the main pass (no barrier between them: each tile is written and counted by one of the two).
// No per-pixel trust and no hold: where the two taps disagree (occlusion, a flash) the blend morphs between them; a real frame is never
// shown locally (the zero vector only where both frames AGREE; Quality's one-sided blend still takes its tap along the motion). The only
// real-frame outputs are whole-frame decisions, uniform over the dispatch: a scene cut (mean finest SAD > cutSad, known from the
// estimate) or a flick (more than globalFrac of the frame untrusted) present the nearer real frame (prev for
// t <= 0.5, curr after), as fallback.comp does for V1; and a blur sub-frame at t <= 0 / t >= 1 is the real frame prev / curr
// (blur_resolve.comp's rule).
// The flick count needs the whole frame, so (as blur_resolve.comp does) the main pass (fix = 0) assumes no flick and counts: an 8x8
// tile (= this workgroup) is untrusted when the mean of its per-pixel mismatch max-channel |warped prev - warped curr| exceeds
// trustMid (V1: a 7x7 sliding mean past the trust ramp's midpoint); its valid pixels go to u[cnt + k]. The fix-up pass (fix = 1, one
// workgroup per 64x64 region) exits at once unless some sample flicked, and then redoes the output with the final counts.
#include "common.glsl"
#define V2_RO readonly restrict   // (the synthesis only reads V2's buffers)
#include "v2_motion.glsl"
#ifdef V2_QUALITY
#include "v2_quality.glsl"
#endif
layout(local_size_x = 8, local_size_y = 8) in;
shared uint esum[16];
const float V2_Z0 = 8.0, V2_Z1 = 12.0, V2_ZMAX = 6.0;
// the block-vector candidates where the taps differ by more than V2_C0 levels; one takes over from V2_C1 levels better (per pixel,
// summed over its 4x4 group), fully at V2_C1 + V2_C2
const float V2_C0 = 24.0, V2_C1 = 8.0, V2_C2 = 16.0;
// full-resolution structure agreement (the static-mask confirmation): ratio of the gradient mismatch to the gradients
const float V2_SEPS = 8.0, V2_SR0 = 0.3, V2_SR1 = 0.6;
const float V2_SZMAX = 12.0;   // ... or the zero-vector mismatch of the pixel and a line of its neighbours (levels; lossy video adds noise)

// A tap on a static overlay picks the overlay up at the wrong place (a dragged copy of HUD text) while the field moves: the tap takes
// the other one (as ui_reject.glsl does with CrosshairV2's mask), softly by how much more of the overlay (za, zc: sOverAt of the
// taps) it hits. Returns x: how much of the overlay the taps leave (1 - the overlay at the tap that hits it least; 0 at the zero
// vector): seen from a pixel inside an overlay (E), a tap outside brings the untinted scene in through a semi-transparent panel;
// y: both taps on the overlay (the overlay at the tap that hits it least): the content of this pixel is hidden in both frames and
// either tap would paste a copy of the overlay; the overlay's own place (the zero vector) is the better guess.
vec2 sReject(vec2 pa, vec2 pc_, inout vec3 wa, inout vec3 wc, float za, float zc) {
    if (all(lessThan(abs(pc_ - pa), vec2(1.0)))) return vec2(0.0);   // (nearly) the zero vector: both taps are the overlay's own place
    vec3 a = wa;
    wa = mix(wa, wc, clamp(2.0 * (za - zc), 0.0, 1.0));
    wc = mix(wc, a, clamp(2.0 * (zc - za), 0.0, 1.0));
    return vec2(1.0 - min(za, zc), min(za, zc));
}

// A tap outside the frame reads the clamped edge (a smeared copy of the border column / row): where content enters or leaves the
// view, only the other frame shows it, so that tap takes the other one (softly over V2_OUT px past the edge). Both outside: as is.
const float V2_OUT = 2.0;
float outside(vec2 p) {
    vec2 o = max(-0.5 - p, p - vec2(pc.W, pc.H) + 0.5);
    return clamp(max(o.x, o.y) * (1.0 / V2_OUT), 0.0, 1.0);
}
void edgeReject(vec2 pa, vec2 pc_, inout vec3 wa, inout vec3 wc) {
    float oa = outside(pa), oc = outside(pc_);
    vec3 a = wa;
    wa = mix(wa, wc, clamp(oa - oc, 0.0, 1.0));
    wc = mix(wc, a, clamp(oc - oa, 0.0, 1.0));
}

// The sum of x over the 4x4 pixels of this lane's group (8x8 workgroup tile, lanes in row order; subgroups of >= 32 lanes)
// (subgroups of fewer lanes: the pixel's own value x 16, a per-pixel choice)
float sum16(float x) {
    if (gl_SubgroupSize < 32u) return 16.0 * x;
    x += subgroupShuffleXor(x, 1u);
    x += subgroupShuffleXor(x, 2u);
    x += subgroupShuffleXor(x, 8u);
    return x + subgroupShuffleXor(x, 16u);
}

// One generated sub-frame at time t (0 < t < 1): 8-bit levels, unrounded; al = the taps' alpha, blended alike; e = the mismatch.
vec3 morph(ivec2 p, float t, out float al, out uint e) {
    vec2 q = vec2(p);
#ifdef V2_STATIC
    uint sbits = sCellBits(sCell(q));   // is an overlay near this pixel (bit 0) or its taps (bit 1)
#else
    const uint sbits = 0u;   // the main pass: far from every overlay (the static pass redoes the cells near one)
#endif
#ifdef V2_QUALITY
#define V2_HV hvQ   // the per-pixel field
#else
#define V2_HV hvAt  // the block field
#endif
    vec2 g = V2_HV(q);
    if (t != 0.5) {
        float s = 2.0 * t - 1.0;
        for (int k = 0; k < 2; k++) g = V2_HV(q - s * g);
    }
    vec2 pa = q - (2.0 * t) * g;
    vec2 pc_ = q + (2.0 - 2.0 * t) * g;
    ivec2 cla = sCell(pa), clc = sCell(pc_);
    bool nearT = (sbits & 2u) != 0u;   // can a tap of this pixel's cell land near an overlay at all (v2_enclose.comp)
    vec4 a = tapRaw(texPrev, pa), c = tapRaw(texCurr, pc_);
    vec3 ra = a.rgb, rc = c.rgb;
    uiReject(pa, pc_, ra, rc);
    vec3 wa = FG_FETCH(ra), wc = FG_FETCH(rc);
    edgeReject(pa, pc_, wa, wc);   // a tap outside the frame takes the other tap
    // a tap on a static overlay (HUD text) takes the other tap. Far from every overlay (most of a frame: whole subgroups, a uniform
    // branch) the taps leave no overlay: sout = 1 (0 at the zero vector), sboth = 0
    float sout = all(lessThan(abs(pc_ - pa), vec2(1.0))) ? 0.0 : 1.0, sboth = 0.0;
    if (subgroupAny(nearT)) {
        vec2 so = sReject(pa, pc_, wa, wc, sOverAt(pa, cla), sOverAt(pc_, clc));
        sout = so.x;
        sboth = so.y;
        if (subgroupAny(sboth > 0.0)) {   // (not where both taps are on CrosshairV2's UI: its patch and its own tap rule decide there)
            bool ha, hc;
            uiRejectHits(pa, pc_, ha, hc);
            if (ha && hc) sboth = 0.0;
        }
    }
    vec3 d = abs(wa - wc);
    float em = max(d.r, max(d.g, d.b));
    vec2 sp = vec2(0.0);
    if (subgroupAny((sbits & 1u) != 0u)) sp = sAt(q, sbits);   // the static-structure mask S and its enclosure E here
    // Where the taps disagree badly: the block vectors around the content's place, each as one rigid motion. Between two
    // blocks whose vectors differ a lot (a periodic texture matched a period apart, an outlier block, a motion edge) the bilinear
    // blend of them is neither motion and its taps sweep the content into ring-shaped outlines; one of the blocks' own vectors often
    // matches there. The choice is made per 4x4 pixels (lanes x ^ 1, 2, y ^ 1, 2 of the 8x8 workgroup tile; at t = 0.5 such a group
    // never straddles a change of the 4 block vectors) by the summed mismatch, so the content moves coherently (no per-pixel specks),
    // and it takes over by how clearly it matches better (soft); where none does, as before. Quality's edge-aware per-pixel field has
    // the same blend between such blocks (fainter), so both modes do it.
    if (subgroupAny(em > V2_C0)) {
        vec2 xm = q - (2.0 * t - 1.0) * g;   // the content's midpoint location
        float S = float(pc.fscale);
        ivec2 i0 = ivec2(floor((xm + 0.5) * (1.0 / (float(V2_BLOCK) * S)) - 0.5));
        float s0 = sum16(em), sb = s0;
        vec3 ba = wa, bc = wc;
        float baa = a.a, bca = c.a;
        vec2 bso = vec2(sout, sboth);   // the overlay test of the winner's taps (sReject's), for the zero-vector test below
        for (int k = 0; k < 4; k++) {
            vec2 gk = S * VF(i0 + ivec2(k & 1, k >> 1));
            vec2 qa = q - (2.0 * t) * gk, qc = q + (2.0 - 2.0 * t) * gk;
            vec4 xa = tapRaw(texPrev, qa), xc = tapRaw(texCurr, qc);
            vec3 ya = xa.rgb, yc = xc.rgb;
            uiReject(qa, qc, ya, yc);
            ya = FG_FETCH(ya); yc = FG_FETCH(yc);
            edgeReject(qa, qc, ya, yc);
            vec2 sok = vec2(all(lessThan(abs(qc - qa), vec2(1.0))) ? 0.0 : 1.0, 0.0);
            if (subgroupAny(nearT)) {   // (the static pass, near an overlay: its tap rejection as for the main taps)
                ivec2 ka = sCell(qa), kc = sCell(qc);
                sok = sReject(qa, qc, ya, yc, sOverAt(qa, ka), sOverAt(qc, kc));
            }
            vec3 dk = abs(ya - yc);
            float sk = sum16(max(dk.r, max(dk.g, dk.b)));   // (the same for the group's 16 lanes: a uniform choice in the group)
            if (sk < sb) { sb = sk; ba = ya; bc = yc; baa = xa.a; bca = xc.a; bso = sok; }
        }
        // (not on an overlay or inside one, S / E: a rigid vector there pastes the scene behind a semi-transparent panel in blocks)
        float wb = clamp((s0 - sb - 16.0 * V2_C1) * (1.0 / (16.0 * V2_C2)), 0.0, 1.0) * (1.0 - clamp(2.0 * max(sp.x, sp.y), 0.0, 1.0));
        vec3 da = abs(ba - bc);
        wa = mix(wa, ba, wb); wc = mix(wc, bc, wb);
        a.a = mix(a.a, baa, wb); c.a = mix(c.a, bca, wb);
        em = mix(em, max(da.r, max(da.g, da.b)), wb);
        sout = mix(sout, bso.x, wb);
        sboth = mix(sboth, bso.y, wb);
    }
#ifdef V2_QUALITY
    vec3 cq = qCovAt(q);   // this pixel's coverage word (its z: the static-overlay mask)
    float wt = qWeightC(t == 0.5 ? cq : qCovAt(q - (2.0 * t - 1.0) * g), t);   // the visible side: curr's share of the blend
#else
    float wt = t;
#endif
    al = (1.0 - wt) * a.a + wt * c.a;
    vec3 m = (1.0 - wt) * wa + wt * wc;
#ifdef V2_QUALITY
    // where the taps still disagree (an occlusion seam, content the field does not follow): a blur along the motion, 4 more tap pairs
    // over +-Q_BLEN of it, so a seam reads as motion blur instead of a hard double edge
    if (em > Q_B0) {
        float beta = clamp((em - Q_B0) * (1.0 / Q_B1), 0.0, 1.0);
        vec2 st = (Q_BLEN * 2.0) * g;   // half-length of the streak (2 g = the motion between the real frames)
        vec3 acc = m;
        for (int j = -2; j <= 2; j++) {
            if (j == 0) continue;
            vec2 o = st * (float(j) * 0.5);
            vec2 qa = pa + o, qc = pc_ + o;
            vec3 xa = tapRaw(texPrev, qa).rgb, xc = tapRaw(texCurr, qc).rgb;
            uiReject(qa, qc, xa, xc);
            vec3 ya = FG_FETCH(xa), yc = FG_FETCH(xc);
            edgeReject(qa, qc, ya, yc);
            ivec2 ka = sCell(qa), kc = sCell(qc);
            if (subgroupAny(sNear(ka) || sNear(kc))) sReject(qa, qc, ya, yc, sOverAt(qa, ka), sOverAt(qc, kc));
            acc += (1.0 - wt) * ya + wt * yc;
        }
        m = mix(m, acc * 0.2, beta);
    }
#endif
    // the zero vector where it fits clearly better: a pixel the two frames show unchanged (a HUD element, text, static background
    // next to a moving object, which the smooth field would drag along) keeps its place
#ifdef V2_QUALITY
    // Quality: its field matches the moving background well, so over a static overlay (subtitle text over a moving scene) both taps
    // can agree on the background and em is small: a pixel with strong local structure counts as disagreeing that much
    float emz = em < V2_Z0 + V2_Z1 + V2_ZMAX ? max(em, qStructure(p)) : em;
#else
    float emz = em;
#endif
    if (emz > V2_Z0 || sp.x > 0.0 || sp.y * sout > 0.0 || sboth > 0.0) {   // (only then can it win)
        vec4 s0 = texelFetch(texPrev, p, 0), s1 = texelFetch(texCurr, p, 0);
        vec3 sa = FG_FETCH(s0.rgb), sc = FG_FETCH(s1.rgb);
        vec3 dz = abs(sa - sc);
        float z = max(dz.r, max(dz.g, dz.b));
        // ... and so do both neighbours along x or along y (a stroke of text or of a HUD element): a lone pixel whose colours happen to
        // match in a moving texture does not switch (it would be a speck). The same neighbours give the full-resolution structure
        // agreement (v2_static.comp's measure, max-channel): the gradients of prev and curr at p against them
        float zn[4];
        float num = 0.0, den = 0.0;
        const bool inS = subgroupAny(sp.x > 0.0);   // (the structure agreement only matters inside the mask; uniform)
        for (int k = 0; k < 4; k++) {
            ivec2 o = k < 2 ? ivec2(2 * k - 1, 0) : ivec2(0, 2 * k - 5);
            ivec2 pn = clamp(p + o, ivec2(0), ivec2(pc.W - 1, pc.H - 1));
            vec3 an = FG_FETCH(texelFetch(texPrev, pn, 0).rgb), cn = FG_FETCH(texelFetch(texCurr, pn, 0).rgb);
            vec3 d = abs(an - cn);
            zn[k] = max(d.r, max(d.g, d.b));
            if (inS) {
                vec3 ga = sa - an, gc = sc - cn;
                vec3 gd = abs(ga - gc), gm = max(abs(ga), abs(gc));
                num += max(gd.r, max(gd.g, gd.b));
                den += max(gm.r, max(gm.g, gm.b));
            }
        }
        float zh = max(zn[0], zn[1]), zv = max(zn[2], zn[3]);
        float zs = max(z, min(zh, zv));
        float ws = clamp((emz - zs - V2_Z0) * (1.0 / V2_Z1), 0.0, 1.0) * clamp(1.0 - zs * (1.0 / V2_ZMAX), 0.0, 1.0);
        // not next to CrosshairV2's UI (its own patch and tap rejection handle it; a static crosshair would vouch for the pixels around it)
        if (ws > 0.0 && (uiHitA(p - 1, p + 1) || uiHitC(p - 1, p + 1))) ws = 0.0;
        // a static structure (the mask), confirmed at full resolution: the pixel and its neighbours agree, or their structure does
        // (a semi-transparent overlay). A thin moving object inside a masked flow pixel fails both and keeps the morph.
        if (inS && sp.x > 0.0) {
            float rf = num / (den + V2_SEPS);
            // On CrosshairV2's UI (its mask and grown rim; the patch pastes only what stayed within its still threshold) only the
            // plain agreement: next to an opaque crosshair the structure test is fooled by a slowly changing background (the big edge
            // step to the crosshair dominates the ratio) and would cross-fade the rim.
            bool onUi = uiHitA(p, p) || uiHitC(p, p);
            float conf = max(onUi ? 0.0 : clamp((V2_SR1 - rf) * (1.0 / (V2_SR1 - V2_SR0)), 0.0, 1.0),
                             clamp(1.0 - zs * (1.0 / V2_SZMAX), 0.0, 1.0));
            ws = max(ws, sp.x * conf);
        }
        // inside an overlay (E, its tinted background between text and border) whose taps leave it: the scene they fetch is not
        // behind the tint; the overlay's own place is (the zero vector)
        ws = max(ws, sp.y * sout);
        ws = max(ws, sboth);   // both taps on the overlay
        m = mix(m, (1.0 - t) * sa + t * sc, ws);
        al = mix(al, (1.0 - t) * s0.a + t * s1.a, ws);
        em = mix(em, z, ws);
    }
    e = uint(em);
    return m;
}

// The output pixel p. counting: add each interior sample's mismatch to esum[k] (all invocations of the workgroup call it, p inside or
// not: subgroup adds), flicks unknown (assumed none); else the flicks are read from the final counts.
void pixelOut(ivec2 p, bool counting, bool cut) {
    bool inside = p.x < pc.W && p.y < pc.H;
    float limit = pc.globalFrac * float(pc.W) * float(pc.H);
    vec3 acc = vec3(0.0);
    float accA = 0.0, ws = 0.0;
    for (int k = 0; k < pc.n; k++) {
        float t = pc.t[k];
        // midpoint rule: slice centres map to x = (2k + 1 - N) / N, the window ends to x = -+1 (blur_resolve.comp)
        float x = (2.0 * float(k) + 1.0 - float(pc.n)) / float(pc.n);
        float w = pc.gauss != 0 ? exp(-2.0 * x * x) : 1.0;
        bool real = t <= 0.0 || t >= 1.0;
        bool fb = cut || (!counting && float(u[pc.cnt + k]) > limit);   // uniform
        vec3 v = vec3(0.0);
        float a = 0.0;
        uint e = 0u;
        if (inside) {
            if (real || fb) {   // a real frame: the nearer one (t <= 0: prev, t >= 1: curr)
                vec4 r = t > 0.5 ? texelFetch(texCurr, p, 0) : texelFetch(texPrev, p, 0);
                v = FG_FETCH(r.rgb);
                a = r.a;
            } else {
                v = clamp(floor(morph(p, t, a, e) + 0.5), 0.0, 255.0);   // the 8-bit sub-frame recordSynth(t) stores
            }
        }
        if (counting && !real && !cut) {   // uniform
            uint s = subgroupAdd(e);
            if (subgroupElect()) atomicAdd(esum[k], s);
        }
        acc += w * v;
        accA += w * a;
        ws += w;
    }
    if (inside) imageStore(outImg, p, FG_STORE(acc / ws, accA / ws));
}

void main() {
    int lid = int(gl_LocalInvocationIndex);
    float meanSad = float(u[U_V2SAD_]) * (1.0 / 256.0) / float(pc.bw * pc.bh);
    bool cut = meanSad > pc.cutSad;
    if (pc.fix == 0) {
        // the tiles near a static overlay (v2_enclose.comp's C of the tile's cell; a tile lies in one cell) are the V2_STATIC
        // variant's, the others the plain one's: two dispatches over the whole frame with no barrier between them (each tile is
        // written by one of them), the plain one without any of the overlay's tests (most of a frame: they cost it nothing)
        bool nearTile = sCellBits(sCell(vec2(ivec2(gl_WorkGroupID.xy) * 8))) != 0u;
#ifdef V2_STATIC
        if (!nearTile) return;
#else
        if (nearTile) return;
#endif
        if (lid < 16) esum[lid] = 0u;
        barrier();
        pixelOut(ivec2(gl_GlobalInvocationID.xy), true, cut);
        barrier();
        if (lid < pc.n && !cut) {
            ivec2 o = ivec2(gl_WorkGroupID.xy) * 8;
            ivec2 nv = min(ivec2(8), ivec2(pc.W, pc.H) - o);
            uint valid = uint(nv.x * nv.y);
            if (float(esum[lid]) > pc.trustMid * float(valid)) atomicAdd(u[pc.cnt + lid], valid);
        }
        return;
    }
    // fix-up: only if some interpolated sample flicked (and no cut, which the main pass already applied)
    if (cut) return;
    bool flicked = false;
    float limit = pc.globalFrac * float(pc.W) * float(pc.H);
    for (int k = 0; k < pc.n; k++)
        if (pc.t[k] > 0.0 && pc.t[k] < 1.0 && float(u[pc.cnt + k]) > limit) flicked = true;
    if (!flicked) return;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            pixelOut(ivec2(gl_WorkGroupID.xy) * 64 + ivec2(i, j) * 8 + ivec2(gl_LocalInvocationID.xy), false, false);
}
