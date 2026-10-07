// V2: the three-class resolve of one 16 x 16 tile (D14), shared by v2_resolve.comp (recordSynth) and v2_blur_resolve.comp (each
// sub-frame of recordSynthBlur), so a blurred sub-frame is the same computation as the synth at its t. The includer defines:
//   V2R_E(p)                    uint mismatch e of the field at t at full-res pixel p (replicate border: p may lie outside)
//   V2R_HV(p)                   the full-res half-vector of the field at t at p
//   V2R_MIS(p, hv, wa, wc)      motion_v1.glsl's mismatch() with the half-vector hv: the two warped taps, returns e
//   V2R_S, V2R_WA, V2R_WC       2t - 1, 1 - t, t
//   V2R_TRUST0, V2R_TRUSTSCALE  the held class's ramp; V2R_OCC0, V2R_OCCSCALE the occlusion ramp; V2R_PICKBIAS; V2R_ZEROBONUS
//                               (Settings::hudBonus)
//   occAt(p, hv, s)             the occlusion map entry of pixel p (motion_v2.glsl)
//   W_, H_, FSCALE_             the full resolution and the flow scale
//
// Per pixel p, with e7 = 7x7 mean of e (V1's box) and Ut = ramp(e7) (V1's trust):
//   no clean occlusion at p (occlusion map side 0):  out = mix(blend, nearer real frame unwarped, Ut)   == V1 (trusted / held)
//   clean occlusion (side a or c, vectors vB, vF):   four 3x3 mismatch means, of the field's vector (e0), of vB (eb), of vF (ef) and
//     of zero (ez), each pixel of the 3x3 with its own vectors. The best of them (another than the field's only if lower by more than
//     pickBias; zero with the hudBonus as in pixfield.comp) gives the two-sided blend Mk and its mismatch ek; w = occlusion ramp(ek);
//     out = mix(Mk, one-sided warp with vB from the side (a: warped prev, c: warped curr), w).
//   So a pixel that one of the three vectors explains is blended with it (a foreground pixel the field gave the background's vector
//   gets the foreground's back: no background over the character; static HUD text between two moving sides keeps zero), and a pixel
//   none explains (the occluded strip) shows the
//   background from the one frame that sees it instead of V1's held patch. The 3x3 instead of V1's 7x7 keeps the occlusion classes to
//   the strip itself. At the edge of a clean occlusion region the result fades into V1's by the share of the 3x3 that has one.
// The untrusted count of the whole-frame fallback is V1's (Ut > 0.5), so its decisions and the bench's fallback% compare with V1.

const int TS = 16, AP = 3, TT = TS + 2 * AP, TB = TS + 2;
shared float te[TT * TT];   // e of the tile + 3 (7x7 box)
shared float hs[TT * TS];   // horizontal 7-sums
shared float tb[TB * TB], tf[TB * TB];   // e of vB / vF of the tile + 1 (= e where the pixel has no clean occlusion)
shared float tz[TB * TB];                 // e of the zero vector (= e where the pixel has no clean occlusion)
shared float tg[TB * TB];                 // 1 = the pixel has a clean occlusion

struct V2Px {
    vec3 o;          // the output (levels, unrounded)
    float alpha;     // of the nearer real frame
    uint cls;        // the class-map byte (motion_v2.glsl)
    bool untrusted;  // Ut > 0.5 (whole-frame fallback count)
};

vec3 v2Blend(vec3 wa, vec3 wc) {
    precise vec3 m = V2R_WA * wa + V2R_WC * wc;
    return m;
}

// Every thread of the workgroup must call it (barriers). p = tile * TS + gl_LocalInvocationID.xy; inside = p in the frame.
V2Px v2ResolvePixel(ivec2 tile, bool nearC) {
    int lid = int(gl_LocalInvocationIndex);
    ivec2 t0 = tile * TS - AP;
    barrier();   // the previous use of the shared arrays is finished
    for (int i = lid; i < TT * TT; i += TS * TS)
        te[i] = float(V2R_E(t0 + ivec2(i % TT, i / TT)));
    barrier();
    for (int i = lid; i < TT * TS; i += TS * TS) {   // horizontal 7-sums: TT rows x TS cols
        int r = i / TS, c = i % TS;
        float s = 0.0;
        for (int k = 0; k < 7; k++) s += te[r * TT + c + k];
        hs[i] = s;
    }
    ivec2 b0 = tile * TS - 1;
    for (int i = lid; i < TB * TB; i += TS * TS) {   // the tile + 1: e of the two side vectors where there is a clean occlusion
        ivec2 q = clamp(b0 + ivec2(i % TB, i / TB), ivec2(0), ivec2(W_ - 1, H_ - 1));
        float e = te[(i / TB + AP - 1) * TT + i % TB + AP - 1];
        float eb = e, ef = e, ez = e;
        uvec4 oc = occAt(q, V2R_HV(q), V2R_S);
        if (oc.z != 0u) {
            vec3 wa, wc;
            eb = float(V2R_MIS(q, float(FSCALE_) * unpackHalf2x16(oc.x), wa, wc));
            ef = float(V2R_MIS(q, float(FSCALE_) * unpackHalf2x16(oc.y), wa, wc));
            ez = float(V2R_MIS(q, vec2(0.0), wa, wc));
        }
        tb[i] = eb;
        tf[i] = ef;
        tz[i] = ez;
        tg[i] = oc.z != 0u ? 1.0 : 0.0;
    }
    barrier();
    ivec2 l = ivec2(gl_LocalInvocationID.xy);
    ivec2 p = tile * TS + l;
    V2Px r;
    r.o = vec3(0.0); r.alpha = 0.0; r.cls = 0u; r.untrusted = false;
    if (p.x >= W_ || p.y >= H_) return r;
    float s7 = 0.0;
    for (int k = 0; k < 7; k++) s7 += hs[(l.y + k) * TS + l.x];
    float ut = clamp((s7 * (1.0 / 49.0) - V2R_TRUST0) * V2R_TRUSTSCALE, 0.0, 1.0);
    r.untrusted = ut > 0.5;
    vec2 hv = V2R_HV(p);
    vec3 wa, wc;
    V2R_MIS(p, hv, wa, wc);
    vec3 m0 = v2Blend(wa, wc);
    vec4 nr = nearC ? texelFetch(texCurr, p, 0) : texelFetch(texPrev, p, 0);
    r.alpha = nr.a;
    uvec4 oc = occAt(p, hv, V2R_S);
    vec3 a8 = FG_FETCH(nr.rgb);
    vec3 v1o = m0 * (1.0 - ut) + a8 * ut;   // trusted / held: V1
    if (oc.z == 0u) {
        r.o = v1o;
        r.cls = (ut > 0.0 ? V2_CLS_HELD : V2_CLS_TRUSTED) | (uint(ut * 15.0 + 0.5) << 4);
        return r;
    }
    // clean occlusion
    float e0 = 0.0, eb = 0.0, ef = 0.0, ez = 0.0, g = 0.0;
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++) {
            e0 += te[(l.y + y + AP - 1) * TT + l.x + x + AP - 1];
            eb += tb[(l.y + y) * TB + l.x + x];
            ef += tf[(l.y + y) * TB + l.x + x];
            ez += tz[(l.y + y) * TB + l.x + x];
            g += tg[(l.y + y) * TB + l.x + x];
        }
    e0 *= 1.0 / 9.0; eb *= 1.0 / 9.0; ef *= 1.0 / 9.0; ez *= 1.0 / 9.0;
    vec2 vB = float(FSCALE_) * unpackHalf2x16(oc.x), vF = float(FSCALE_) * unpackHalf2x16(oc.y);
    vec3 ba, bc;
    V2R_MIS(p, vB, ba, bc);
    uint k = V2_CLS_OCC_FIELD;
    float ek = e0;
    vec3 mk = m0;
    if (eb + V2R_PICKBIAS < ek) { ek = eb; k = V2_CLS_OCC_BG; mk = v2Blend(ba, bc); }
    if (ef + V2R_PICKBIAS < ek) {
        ek = ef; k = V2_CLS_OCC_FG;
        vec3 fa, fc;
        V2R_MIS(p, vF, fa, fc);
        mk = v2Blend(fa, fc);
    }
    if (ez - V2R_ZEROBONUS + V2R_PICKBIAS < ek) {
        ek = max(ez - V2R_ZEROBONUS, 0.0); k = V2_CLS_OCC_ZERO;
        vec3 za, zc;
        V2R_MIS(p, vec2(0.0), za, zc);
        mk = v2Blend(za, zc);
    }
    float w = clamp((ek - V2R_OCC0) * V2R_OCCSCALE, 0.0, 1.0);
    vec3 one = oc.z == 1u ? ba : bc;   // covered: prev sees it; uncovered: curr
    // at the edge of an occluded region (some 3x3 neighbours without a clean occlusion) fade to V1's output: no hard seam between
    // the classes
    g *= 1.0 / 9.0;
    vec3 oo = mk * (1.0 - w) + one * w;
    r.o = v1o * (1.0 - g) + oo * g;
    r.cls = k | (oc.z == 2u ? 8u : 0u) | (uint(w * 15.0 + 0.5) << 4);
    return r;
}
