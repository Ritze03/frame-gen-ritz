// V2 Performance synthesis ("grid morph"), shared by v2/group.frag.glsl and v2/morph.frag.glsl (the host assembles
// common.glsl + this file + the program file). Port of v2_morph.glsl + v2_motion.glsl (frame-gen-ritz framegen-v2 @ 273bbad), Performance
// mode, v2StaticMask on/off through the mask's content, no CrosshairV2 / blur (their hooks are no-ops with UI protection off).
//
// One pixel at time t: g = the cleaned block field upsampled bilinearly (hvAt), taken to time t by a 2-step fixed point;
// out = (1 - t) prev(q - 2t g) + t curr(q + 2(1 - t) g), both taps bilinear (levels, rounded); a tap outside the frame takes the other
// tap; taps on a static overlay take the other tap (sReject); where the taps disagree badly one of the 4 block vectors around may take
// over per 4x4 pixel group (v2/group.frag.glsl decides the group's winner); the zero vector where the frames agree and the taps do not.
uniform sampler2D uPrev;     // the frames (RGBA8, or RGBA16F with FG_F16)
uniform sampler2D uCurr;
uniform sampler2D uVf;       // RG32F cleaned block field (half-vectors in flow pixels)
uniform sampler2D uSmask;        // R32F static-structure mask, flow resolution
uniform sampler2D uEC;       // RG32F per 8x8 flow cell: E (enclosure), C (bit 0 overlay near, bit 1 taps may land near one)
uniform sampler2D uEm;       // R32F per pixel: the taps' mismatch em (v2/em), read by v2/group and v2/morph
uniform ivec2 uFrameSize;
uniform int uFlowScale;
uniform float uT;

const int V2_BLOCK = 8;
const float V2_Z0 = 8.0, V2_Z1 = 12.0, V2_ZMAX = 6.0;
const float V2_C0 = 24.0, V2_C1 = 8.0, V2_C2 = 16.0;
const float V2_SEPS = 8.0, V2_SR0 = 0.3, V2_SR1 = 0.6;
const float V2_SD0 = 16.0, V2_SD1 = 24.0;
const float V2_SZMAX = 12.0;
const float V2_OUT = 2.0;

vec2 VF(ivec2 b) { return texelFetch(uVf, clamp(b, ivec2(0), textureSize(uVf, 0) - 1), 0).xy; }

// the block field upsampled bilinearly straight to full resolution (x: pixel-centre coordinates), x fscale
vec2 hvAt(vec2 x) {
    float S = float(uFlowScale);
    vec2 s = (x + 0.5) * (1.0 / (float(V2_BLOCK) * S)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    return S * mix(mix(VF(i), VF(i + ivec2(1, 0)), f.x), mix(VF(i + ivec2(0, 1)), VF(i + ivec2(1, 1)), f.x), f.y);
}

float sF(ivec2 m) { return texelFetch(uSmask, clamp(m, ivec2(0), textureSize(uSmask, 0) - 1), 0).r; }
float eF(ivec2 c) { return texelFetch(uEC, clamp(c, ivec2(0), textureSize(uEC, 0) - 1), 0).r; }
ivec2 sCell(vec2 x) {
    return clamp(ivec2(floor((x + 0.5) * (1.0 / float(V2_BLOCK * uFlowScale)))), ivec2(0), textureSize(uEC, 0) - 1);
}
uint sCellBits(ivec2 c) { return uint(texelFetch(uEC, c, 0).g + 0.5); }
bool sNear(ivec2 c) { return (sCellBits(c) & 1u) != 0u; }
ivec2 sTexel(vec2 x) { return ivec2(floor((x + 0.5) * (1.0 / float(uFlowScale)))); }

// S at the nearest flow pixel (x), E bilinear on the cell grid, and S dilated by ~1 flow px (largest of the 4 S texels around x)
vec3 sAt(vec2 x, uint bits) {
    if ((bits & 1u) == 0u) return vec3(0.0);
    vec2 s = (x + 0.5) * (1.0 / float(V2_BLOCK * uFlowScale)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    float E = mix(mix(eF(i), eF(i + ivec2(1, 0)), f.x), mix(eF(i + ivec2(0, 1)), eF(i + ivec2(1, 1)), f.x), f.y);
    ivec2 m = ivec2(floor((x + 0.5) * (1.0 / float(uFlowScale)) - 0.5)), n = sTexel(x);
    float s00 = sF(m), s10 = sF(m + ivec2(1, 0)), s01 = sF(m + ivec2(0, 1)), s11 = sF(m + ivec2(1, 1));
    float sn = n.y == m.y ? (n.x == m.x ? s00 : s10) : (n.x == m.x ? s01 : s11);
    return vec3(sn, E, max(max(s00, s10), max(s01, s11)));
}
// the overlay at x for the tap rejection: S dilated by ~1 flow px, and E of the cell c = sCell(x)
vec2 sOverAt(vec2 x, ivec2 c) {
    if (!sNear(c)) return vec2(0.0);
    ivec2 i = ivec2(floor((x + 0.5) * (1.0 / float(uFlowScale)) - 0.5));
    return vec2(max(max(sF(i), sF(i + ivec2(1, 0))), max(sF(i + ivec2(0, 1)), sF(i + ivec2(1, 1)))), eF(c));
}

// A bilinear tap of a frame at p, 1/32 px quantised, as working values (Rgba8: rounded to integer levels = FG_FETCH(textureLod(...))).
vec3 tap(sampler2D t, vec2 p) { return fgTap(t, p); }

// A tap on a static overlay takes the other tap, softly by how much more of the overlay it hits (sReject of v2_morph.glsl).
vec3 sReject(vec2 pa, vec2 pc_, inout vec3 wa, inout vec3 wc, vec2 za2, vec2 zc2) {
    if (all(lessThan(abs(pc_ - pa), vec2(1.0)))) return vec3(0.0);
    float za = max(za2.x, za2.y), zc = max(zc2.x, zc2.y);
    vec3 a = wa;
    wa = mix(wa, wc, clamp(2.0 * (za - zc), 0.0, 1.0));
    wc = mix(wc, a, clamp(2.0 * (zc - za), 0.0, 1.0));
    return vec3(1.0 - min(za, zc), min(za, zc), min(za2.x, zc2.x));
}
// both taps on S: how surely the structure they hit is static (ra, rc: the taps' levels)
float tapsStatic(vec2 pa, vec2 pc_, vec3 ra, vec3 rc) {
    vec3 d = max(FG_D(ra, tap(uCurr, pa)), FG_D(rc, tap(uPrev, pc_)));
    return clamp(1.0 - max(d.r, max(d.g, d.b)) * (1.0 / V2_SZMAX), 0.0, 1.0);
}
// a tap outside the frame takes the other tap (softly over V2_OUT px past the edge)
float outside(vec2 p) {
    vec2 o = max(-0.5 - p, p - vec2(uFrameSize) + 0.5);
    return clamp(max(o.x, o.y) * (1.0 / V2_OUT), 0.0, 1.0);
}
void edgeReject(vec2 pa, vec2 pc_, inout vec3 wa, inout vec3 wc) {
    float oa = outside(pa), oc = outside(pc_);
    vec3 a = wa;
    wa = mix(wa, wc, clamp(oa - oc, 0.0, 1.0));
    wc = mix(wc, a, clamp(oc - oa, 0.0, 1.0));
}
float maxc(vec3 d) { return max(d.r, max(d.g, d.b)); }

// The library tries the block-vector candidates only if some lane of the SUBGROUP has em > V2_C0 (a uniform branch: subgroupAny). Its
// subgroup is the 8x8 workgroup tile on the reference GPU (RADV RDNA3 compute: wave64); here the tile is the gate, the lanes outside
// the frame do not count (they never run morph()).
bool tileLit(ivec2 p) {
    ivec2 o = (p >> 3) << 3;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++) {
            ivec2 q = o + ivec2(i, j);
            if (q.x < uFrameSize.x && q.y < uFrameSize.y && texelFetch(uEm, q, 0).r > V2_C0) return true;
        }
    return false;
}

#ifdef V2_QUALITY
// ---- V2 Quality (the "coverage morph", gpu/shaders/v2_quality.glsl; the host defines V2_QUALITY for v2/qem, v2/qgroup, v2/qmorph) ----
uniform sampler2D uQf;    // RGBA32F per-pixel field (xy, flow px of half-vector), z = static-overlay mask (v2/qfield)
uniform sampler2D uCov;   // RGBA32F coverage: cP, cC, z, quantised as the library packs them (v2/qcov)
const float Q_EPS = 0.03;
const float Q_B0 = 16.0, Q_B1 = 24.0;
const float Q_BLEN = 0.25;
const float Q_SGAIN = 0.5;

vec2 qF(ivec2 m) { return texelFetch(uQf, clamp(m, ivec2(0), textureSize(uQf, 0) - 1), 0).xy; }
// the per-pixel field (full-res px of half-vector) at full-resolution position x: bilinear, x fscale
vec2 hvQ(vec2 x) {
    float S = float(uFlowScale);
    vec2 s = (x + 0.5) * (1.0 / S) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i = ivec2(fl);
    return S * mix(mix(qF(i), qF(i + ivec2(1, 0)), f.x), mix(qF(i + ivec2(0, 1)), qF(i + ivec2(1, 1)), f.x), f.y);
}
// the coverage bilinear at full-resolution position q: (cP, cC, z)
vec3 qCovAt(vec2 q) {
    ivec2 mx = textureSize(uCov, 0) - 1;
    vec2 m = clamp((q + 0.5) * (1.0 / float(uFlowScale)) - 0.5, vec2(0.0), vec2(mx));
    vec2 fl = floor(m);
    vec2 f = m - fl;
    ivec2 i = ivec2(fl);
    ivec2 j = min(i + 1, mx);
    vec3 c00 = texelFetch(uCov, i, 0).xyz, c01 = texelFetch(uCov, ivec2(j.x, i.y), 0).xyz;
    vec3 c10 = texelFetch(uCov, ivec2(i.x, j.y), 0).xyz, c11 = texelFetch(uCov, j, 0).xyz;
    return mix(mix(c00, c01, f.x), mix(c10, c11, f.x), f.y);
}
// the gather's share of curr at time t for content from the location whose coverage is cv
float qWeightC(vec3 cv, float t) {
    float a = (1.0 - t) * (cv.x + Q_EPS), c = t * (cv.y + Q_EPS);
    return c / (a + c);
}
// local structure at full-res pixel p, levels x Q_SGAIN: the largest max-channel difference of prev's pixel to its right and lower
// neighbours (counts as disagreement of the taps for the zero-vector test)
float qStructure(ivec2 p) {
    ivec2 mx = uFrameSize - 1;
    vec3 c = FG_LV(fgPx(uPrev, p));
    vec3 r = FG_LV(fgPx(uPrev, min(p + ivec2(1, 0), mx)));
    vec3 d = FG_LV(fgPx(uPrev, min(p + ivec2(0, 1), mx)));
    vec3 a = max(abs(c - r), abs(c - d));
    return Q_SGAIN * max(a.r, max(a.g, a.b));
}
#define V2_HV hvQ
#else
#define V2_HV hvAt
#endif

// g: the field (Performance: the cleaned block field; Quality: the per-pixel field) upsampled to q, taken to time t by the 2-step
// fixed point g = h(q - (2t - 1) g)
vec2 gAt(vec2 q, float t) {
    vec2 g = V2_HV(q);
    if (t != 0.5) {
        float s = 2.0 * t - 1.0;
        for (int k = 0; k < 2; k++) g = V2_HV(q - s * g);
    }
    return g;
}

// ---- the pixel's own taps (the part of morph() before the block-vector candidates) ----
struct Front {
    vec2 q, g, pa, pc;
    uint sbits;
    bool nearT;
    vec3 ta, tc;            // raw taps (levels)
    vec3 wa, wc;            // after the frame-edge and overlay rejection
    float sout, sboth, sbS; // sReject's results (sout: 1 - the overlay at the tap that hits it least, 0 at the zero vector)
    float em, em0;          // the taps' mismatch (after / before the overlay rejection)
    vec2 sp;                // S (nearest flow pixel), E (bilinear) at the pixel
    float sxd;              // S dilated by ~1 flow px
};

Front front(ivec2 p, float t) {
    Front F;
    vec2 q = vec2(p);
    F.q = q;
    F.sbits = sCellBits(sCell(q));
    vec2 g = gAt(q, t);
    F.g = g;
    vec2 pa = q - (2.0 * t) * g;
    vec2 pc_ = q + (2.0 - 2.0 * t) * g;
    F.pa = pa; F.pc = pc_;
    F.nearT = (F.sbits & 2u) != 0u;
    vec3 ta = tap(uPrev, pa), tc = tap(uCurr, pc_);
    F.ta = ta; F.tc = tc;
    vec3 wa = ta, wc = tc;
    edgeReject(pa, pc_, wa, wc);
    float sout = all(lessThan(abs(pc_ - pa), vec2(1.0))) ? 0.0 : 1.0, sboth = 0.0, sbS = 0.0;
    F.em0 = maxc(FG_D(wa, wc));
    if (F.nearT) {
        vec3 so = sReject(pa, pc_, wa, wc, sOverAt(pa, sCell(pa)), sOverAt(pc_, sCell(pc_)));
        sout = so.x;
        sboth = so.y;
        sbS = so.z;
        if (sbS > 0.0) sbS *= tapsStatic(pa, pc_, ta, tc);
    }
    F.wa = wa; F.wc = wc;
    F.sout = sout; F.sboth = sboth; F.sbS = sbS;
    F.em = maxc(FG_D(wa, wc));
    F.sp = vec2(0.0);
    F.sxd = 0.0;
    if ((F.sbits & 1u) != 0u) {
        vec3 s3 = sAt(q, F.sbits);
        F.sp = s3.xy;
        F.sxd = s3.z;
    }
    return F;
}

// ---- a block-vector candidate (k = 0..3: the 4 block vectors around the content's place) ----
struct Cand {
    vec3 ya, yc;
    vec3 sok;     // sReject's results for this candidate's taps
    float ek0;    // mismatch before the overlay rejection
    float ek;     // mismatch after
};

Cand candTap(vec2 q, float t, vec2 g, bool nearT, int k) {
    Cand C;
    vec2 xm = q - (2.0 * t - 1.0) * g;   // the content's midpoint location
    float S = float(uFlowScale);
    ivec2 i0 = ivec2(floor((xm + 0.5) * (1.0 / (float(V2_BLOCK) * S)) - 0.5));
    vec2 gk = S * VF(i0 + ivec2(k & 1, k >> 1));
    vec2 qa = q - (2.0 * t) * gk, qc = q + (2.0 - 2.0 * t) * gk;
    vec3 xa = tap(uPrev, qa), xc = tap(uCurr, qc);
    vec3 ya = xa, yc = xc;
    edgeReject(qa, qc, ya, yc);
    vec3 sok = vec3(all(lessThan(abs(qc - qa), vec2(1.0))) ? 0.0 : 1.0, 0.0, 0.0);
    C.ek0 = maxc(FG_D(ya, yc));
    if (nearT) {
        sok = sReject(qa, qc, ya, yc, sOverAt(qa, sCell(qa)), sOverAt(qc, sCell(qc)));
        if (sok.z > 0.0) sok.z *= tapsStatic(qa, qc, xa, xc);
    }
    C.ya = ya; C.yc = yc; C.sok = sok;
    C.ek = maxc(FG_D(ya, yc));
    return C;
}
