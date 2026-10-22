// V2 Performance synthesis, step 2 (full resolution): morph() of v2_morph.glsl for the pixel, n = 1 sample. Writes the 8-bit sub-frame
// (RGBA8; RgbaF16: RGBA16F) and e = uint(em), the pixel's mismatch (R16F, integer levels) for the flick count. The whole-frame scene-cut / flick fallback is v2/final.
uniform sampler2D uGroup;   // v2/group: x = s0, y = sb, z = winner + 1, w = 1 for a group cut by the frame edge
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oE;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float t = uT;
    Front F = front(p, t);
    vec3 wa = F.wa, wc = F.wc;
    float em = F.em, em0 = F.em0, sout = F.sout, sboth = F.sboth, sbS = F.sbS;
    vec2 sp = F.sp;
    float sxd = F.sxd;

    // Where the taps disagree badly: one of the 4 block vectors around the content's place, chosen per 4x4 group (v2/group)
    float s0, sb;
    int win;
    bool full = (p.x | 3) < uFrameSize.x && (p.y | 3) < uFrameSize.y;
    if (full) {
        vec4 G = texelFetch(uGroup, p >> 2, 0);
        s0 = G.x; sb = G.y; win = int(G.z + 0.5);
    } else {   // the library's per-pixel choice at the frame's edge: the pixel's own value x 16
        s0 = 16.0 * em;
        sb = s0;
        win = 0;
        if (tileLit(p)) {
            for (int k = 0; k < 4; k++) {
                float sk = 16.0 * candTap(F.q, t, F.g, F.nearT, k).ek;
                if (sk < sb) { sb = sk; win = k + 1; }
            }
        }
    }
    float wb0 = clamp((s0 - sb - 16.0 * V2_C1) * (1.0 / (16.0 * V2_C2)), 0.0, 1.0);
    if (win > 0 && wb0 > 0.0) {
        Cand C = candTap(F.q, t, F.g, F.nearT, win - 1);
        // (not on an overlay, S, nor inside one, E, where the winner's taps leave it)
        float wb = wb0 * (1.0 - clamp(2.0 * max(sp.x, sp.y * C.sok.x), 0.0, 1.0));
        wa = mix(wa, C.ya, wb); wc = mix(wc, C.yc, wb);
        em = mix(em, maxc(FG_D(C.ya, C.yc)), wb);
        sout = mix(sout, C.sok.x, wb);
        sboth = mix(sboth, C.sok.y, wb);
        sbS = mix(sbS, C.sok.z, wb);
        em0 = mix(em0, C.ek0, wb);
    }
#ifdef V2_QUALITY
    // the visible side: the blend leans to the frame that still shows the content of its midpoint location (curr's share wt)
    float wt = qWeightC(t == 0.5 ? qCovAt(F.q) : qCovAt(F.q - (2.0 * t - 1.0) * F.g), t);
#else
    float wt = t;
#endif
    vec3 m = (1.0 - wt) * wa + wt * wc;
#ifdef V2_QUALITY
    // where the taps still disagree (an occlusion seam, content the field does not follow): a blur along the motion, 4 more tap pairs
    // over +-Q_BLEN of it
    if (em > Q_B0) {
        float beta = clamp((em - Q_B0) * (1.0 / Q_B1), 0.0, 1.0);
        vec2 st = (Q_BLEN * 2.0) * F.g;
        vec3 acc = m;
        for (int j = -2; j <= 2; j++) {
            if (j == 0) continue;
            vec2 o = st * (float(j) * 0.5);
            vec2 qa = F.pa + o, qc = F.pc + o;
            vec3 ya = tap(uPrev, qa), yc = tap(uCurr, qc);
            edgeReject(qa, qc, ya, yc);
            ivec2 ka = sCell(qa), kc = sCell(qc);
            if (sNear(ka) || sNear(kc)) sReject(qa, qc, ya, yc, sOverAt(qa, ka), sOverAt(qc, kc));
            acc += (1.0 - wt) * ya + wt * yc;
        }
        m = mix(m, acc * 0.2, beta);
    }
    // its field matches a moving background well, so over a static overlay both taps can agree on the background: strong local
    // structure counts as that much disagreement for the zero-vector test
    float emz = em < V2_Z0 + V2_Z1 + V2_ZMAX ? max(em, qStructure(p)) : em;
#else
    float emz = em;
#endif

    // the zero vector where it fits clearly better: a pixel the two frames show unchanged (a HUD element, text, static background
    // next to a moving object, which the smooth field would drag along) keeps its place
    if (emz > V2_Z0 || sxd > 0.0 || sp.y * sout > 0.0 || sboth > 0.0) {
        vec3 sa = fgPx(uPrev, p), sc = fgPx(uCurr, p);
        float z = maxc(FG_D(sa, sc));
        float zn[4];
        float num = 0.0, den = 0.0;
        bool inS = sxd > 0.0;
        for (int k = 0; k < 4; k++) {
            ivec2 o = k < 2 ? ivec2(2 * k - 1, 0) : ivec2(0, 2 * k - 5);
            vec3 an = fgPx(uPrev, p + o), cn = fgPx(uCurr, p + o);
            zn[k] = maxc(FG_D(an, cn));
            if (inS) {
                vec3 ga = FG_G(sa, an), gc = FG_G(sc, cn);
                num += maxc(abs(ga - gc));
                den += maxc(max(abs(ga), abs(gc)));
            }
        }
        float zh = max(zn[0], zn[1]), zv = max(zn[2], zn[3]);
        float zs = max(z, min(zh, zv));
        float ws = clamp((emz - zs - V2_Z0) * (1.0 / V2_Z1), 0.0, 1.0) * clamp(1.0 - zs * (1.0 / V2_ZMAX), 0.0, 1.0);
        // a static structure (the mask, dilated by ~1 flow px), confirmed at full resolution: the pixel and its neighbours agree, or
        // their structure does (a semi-transparent overlay) where there is structure at all
        if (inS) {
            float rf = num * (1.0 / (den + V2_SEPS));
            float conf = max(clamp((V2_SR1 - rf) * (1.0 / (V2_SR1 - V2_SR0)), 0.0, 1.0) * clamp((den - V2_SD0) * (1.0 / V2_SD1), 0.0, 1.0),
                             clamp(1.0 - zs * (1.0 / V2_SZMAX), 0.0, 1.0));
            ws = max(ws, sxd * conf);
        }
        // inside an overlay (E) whose taps leave it, and both taps on the overlay: the zero vector, unless the morph's own taps fit
        // clearly better
        float fitM = clamp((zs - em0 - V2_Z0) * (1.0 / V2_Z1), 0.0, 1.0);
        ws = max(ws, sp.y * sout * (1.0 - fitM));
        ws = max(ws, sboth * (1.0 - fitM));
        ws = max(ws, sbS);
        m = mix(m, (1.0 - t) * sa + t * sc, ws);
        em = mix(em, z, ws);
    }
    oColor = fgStore(m);   // Rgba8: the 8-bit sub-frame; RgbaF16: the half-float one
    oE = vec4(float(uint(em)), 0.0, 0.0, 1.0);   // R16F: the integer mismatch (levels)
}
