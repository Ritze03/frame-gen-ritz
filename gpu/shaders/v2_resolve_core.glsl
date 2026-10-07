// V2: the three-class resolve of one 16 x 16 tile (D14), shared by v2_resolve.comp (recordSynth) and v2_blur_resolve.comp (each
// sub-frame of recordSynthBlur), so a blurred sub-frame is the same computation as the synth at its t. The includer defines:
//   V2R_E(p)                    uint mismatch e of the field at t at full-res pixel p (replicate border: p may lie outside);
//                               V2_E_UNKNOWN = a warp tap was on the UI (v2_error.comp)
//   V2R_HV(p)                   the full-res half-vector of the field at t at p
//   V2R_MIS(p, hv, wa, wc)      motion_v1.glsl's mismatch() with the half-vector hv: the two warped taps, returns e
//   V2R_S, V2R_WA, V2R_WC       2t - 1, 1 - t, t
//   V2R_TRUST0, V2R_TRUSTSCALE  the held class's ramp; V2R_OCC0, V2R_OCCSCALE the occlusion ramp; V2R_PICKBIAS; V2R_ZEROBONUS
//                               (Settings::hudBonus)
//   occAt(p, hv, s)             the occlusion map entry of pixel p (motion_v2.glsl); occSideAt(j) its side at flow pixel j (clamped)
//   W_, H_, FSCALE_             the full resolution and the flow scale
//   V2R_FEATHER                 V2Knobs holdFeather (full-resolution px, <= 32; 0 = V1's hold edge), uniform
//   V2R_UF(c, cw)               the soft hold edge's uf of 4 x 4 px cell c (cw cells per row; v2_hold_core.glsl), read if V2R_FEATHER > 0
//   V2R_OCCSOFT                 V2Knobs occSoft (levels; 0 = the hard pick at a clean occlusion), uniform
//
// Per pixel p, with e7 = 7x7 mean of e (V1's box, over the pixels whose e is known: a pixel with a warp tap on the UI (CrosshairV2
// tap rejection) shows the other frame's tap for both, which says nothing about its vector, so it takes the trust of the pixels
// around it; all unknown: e7 = 0) and Ut = ramp(e7) (V1's trust). A pixel whose own e is unknown, or whose box is mostly unknown
// (next to the UI), is also held as far as its vector is a stray one: shared (within 1/4 px) by fewer than 4 of its 8 neighbours
// (fully at 2 or fewer); nothing else checks it there, and a static UI pulls the estimate's vectors around it:
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
// The soft hold edge (I11, D18; V2R_FEATHER > 0): the pixel's hold is max(Ut, uf), uf = the map of v2_hold_core.glsl (the hold
// weight falling off softly outside held regions instead of V1's hard ~4 px edge), also in the class map's weight. The untrusted
// count is that of Ut.

const int TS = 16, AP = 3, TT = TS + 2 * AP, TB = TS + 2;
shared float te[TT * TT];   // e + KN of the tile + 3 (7x7 box); 0 where e is unknown (V2_E_UNKNOWN)
shared float hs[TT * TS];   // horizontal 7-sums
const float KN = 16384.0;   // the count of known pixels rides along in the sums (49 x 254 < KN; 49 x (KN + 254) < 2^24: exact)
shared float tb[TB * TB], tf[TB * TB];   // e of vB / vF of the tile + 1 (= e where the pixel has no clean occlusion)
shared float tz[TB * TB];                 // e of the zero vector (= e where the pixel has no clean occlusion)
shared float tg[TB * TB];                 // 1 = the pixel has a clean occlusion
shared vec2 thv[TB * TB];                 // the half-vector of the field at t

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
    for (int i = lid; i < TT * TT; i += TS * TS) {
        uint e = V2R_E(t0 + ivec2(i % TT, i / TT));
        bool known = e != V2_E_UNKNOWN;
        te[i] = known ? float(e) + KN : 0.0;
    }
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
        float e = max(te[(i / TB + AP - 1) * TT + i % TB + AP - 1] - KN, 0.0);
        float eb = e, ef = e, ez = e;
        vec2 hq = V2R_HV(q);
        thv[i] = hq;
        uvec4 oc = occAt(q, hq, V2R_S);
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
    float F = V2R_FEATHER;
    ivec2 l = ivec2(gl_LocalInvocationID.xy);
    ivec2 p = tile * TS + l;
    V2Px r;
    r.o = vec3(0.0); r.alpha = 0.0; r.cls = 0u; r.untrusted = false;
    if (p.x >= W_ || p.y >= H_) return r;
    float s7 = 0.0;
    for (int k = 0; k < 7; k++) s7 += hs[(l.y + k) * TS + l.x];
    float n7 = floor(s7 * (1.0 / KN));   // known pixels of the 7x7
    s7 -= n7 * KN;
    float e7 = n7 == 49.0 ? s7 * (1.0 / 49.0) : n7 > 0.0 ? s7 / n7 : 0.0;
    float ut = clamp((e7 - V2R_TRUST0) * V2R_TRUSTSCALE, 0.0, 1.0);
    r.untrusted = ut > 0.5;
    float ufe = 0.0;   // the soft hold edge (v2_hold_core.glsl's map, bilinear between the 4 x 4 px cell centres; not counted)
    if (F > 0.0) {
        ivec2 cw = ivec2((W_ + 3) >> 2, (H_ + 3) >> 2);
        vec2 sc = (vec2(p) + 0.5) * 0.25 - 0.5;
        ivec2 i0 = ivec2(floor(sc));
        vec2 f = sc - vec2(i0);
        ivec2 j0 = clamp(i0, ivec2(0), cw - 1), j1 = clamp(i0 + 1, ivec2(0), cw - 1);
        float u00 = V2R_UF(ivec2(j0.x, j0.y), cw.x), u01 = V2R_UF(ivec2(j1.x, j0.y), cw.x);
        float u10 = V2R_UF(ivec2(j0.x, j1.y), cw.x), u11 = V2R_UF(ivec2(j1.x, j1.y), cw.x);
        ufe = mix(mix(u00, u01, f.x), mix(u10, u11, f.x), f.y);
    }
    if (te[(l.y + AP) * TT + l.x + AP] == 0.0 || n7 < 25.0) {
        // e unknown (a tap on the UI: the output is the other frame's tap, right only if p's vector is, which nothing measures) or
        // mostly unknown around p: a vector that most of its 8 neighbours do not share (within 1/4 px) is a stray one (a vector the
        // static UI pulled, or the in-between vector of a field edge next to it): held as far as fewer than 4 agree (fully at 2 or
        // fewer)
        int agree = 0;
        for (int k = 0; k < 9; k++) {
            ivec2 o = ivec2(k % 3 - 1, k / 3 - 1);
            vec2 dv = abs(thv[(l.y + 1 + o.y) * TB + l.x + 1 + o.x] - thv[(l.y + 1) * TB + l.x + 1]);
            agree += k != 4 && max(dv.x, dv.y) <= 0.25 ? 1 : 0;
        }
        ut = max(ut, clamp(0.5 * float(4 - agree), 0.0, 1.0));
    }
    vec2 hv = thv[(l.y + 1) * TB + l.x + 1];   // == V2R_HV(p) (read once above: the field at t of the blur is computed on the fly)
    vec3 wa, wc;
    V2R_MIS(p, hv, wa, wc);
    vec3 m0 = v2Blend(wa, wc);
    vec4 nr = nearC ? texelFetch(texCurr, p, 0) : texelFetch(texPrev, p, 0);
    r.alpha = nr.a;
    uvec4 oc = occAt(p, hv, V2R_S);
    vec3 a8 = FG_FETCH(nr.rgb);
    // the content rescue's pixels (occlusion map w): an object both frames show, put back at its midpoint with a vector that rarely
    // matches exactly (perspective); holding, or the background vector of a clean occlusion, would show the background it covers,
    // i.e. delete it again: the two-sided blend of its own vector
    bool resc = (oc.w & 1u) != 0u;
    if (resc) ut = 0.0;
    float utf = resc ? 0.0 : max(ut, ufe);
    vec3 v1o = m0 * (1.0 - utf) + a8 * utf;   // trusted / held: V1 (with the soft hold edge)
    if (oc.z == 0u || resc) {   // (a rescued pixel is two-sided with its own vector: the occlusion picks would put the background
                                // back over the object)
        r.o = v1o;
        r.cls = (resc ? V2_CLS_RESCUED : utf > 0.0 ? V2_CLS_HELD : V2_CLS_TRUSTED) | (uint(utf * 15.0 + 0.5) << 4);
        return r;
    }
    // clean occlusion
    float e0 = 0.0, eb = 0.0, ef = 0.0, ez = 0.0, g = 0.0;
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 3; x++) {
            e0 += max(te[(l.y + y + AP - 1) * TT + l.x + x + AP - 1] - KN, 0.0);
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
    float sig = V2R_OCCSOFT;
    if (sig > 0.0) {   // soft pick (I11): every candidate by exp(-(its biased cost - the best's) / occSoft), not the best alone
        float c0 = e0, cb = eb + V2R_PICKBIAS, cf = ef + V2R_PICKBIAS, cz = max(ez - V2R_ZEROBONUS, 0.0) + V2R_PICKBIAS;
        float cm = min(min(c0, cb), min(cf, cz));
        float w0 = exp((cm - c0) / sig), wb = exp((cm - cb) / sig), wf = exp((cm - cf) / sig), wz = exp((cm - cz) / sig);
        vec3 fa, fc, za, zc;
        V2R_MIS(p, vF, fa, fc);
        V2R_MIS(p, vec2(0.0), za, zc);
        float ws = w0 + wb + wf + wz;
        mk = (w0 * m0 + wb * v2Blend(ba, bc) + wf * v2Blend(fa, fc) + wz * v2Blend(za, zc)) / ws;
        ek = (w0 * e0 + wb * eb + wf * ef + wz * max(ez - V2R_ZEROBONUS, 0.0)) / ws;
    }
    float w = clamp((ek - V2R_OCC0) * V2R_OCCSCALE, 0.0, 1.0);
    vec3 one = oc.z == 1u ? ba : bc;   // covered: prev sees it; uncovered: curr
    // at the edge of an occluded region (some 3x3 neighbours without a clean occlusion) fade to V1's output: no hard seam between
    // the classes
    g *= 1.0 / 9.0;
    // (I11) an island of clean occlusion (fewer than 15 of the 5x5 flow pixels around p's own: inside a held region, a lighting
    // change, noise) fades to V1's output too: its one-sided warp would be a speck of the other frame in the held patch
    {
        vec2 m = vec2(p) - V2R_S * hv;
        ivec2 j = ivec2(floor((m + 0.5) / float(FSCALE_)));
        int n5 = 0;
        for (int d = 0; d < 25; d++) n5 += occSideAt(j + ivec2(d % 5 - 2, d / 5 - 2)) != 0u ? 1 : 0;
        g *= clamp(float(n5 - 7) * (1.0 / 8.0), 0.0, 1.0);
    }
    vec3 oo = mk * (1.0 - w) + one * w;
    r.o = v1o * (1.0 - g) + oo * g;
    r.cls = k | (oc.z == 2u ? 8u : 0u) | (uint(w * 15.0 + 0.5) << 4);
    return r;
}
