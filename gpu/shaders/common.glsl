// Shared declarations for all framegen passes. Mirrors proto/interp.py (the frozen reference).

layout(set = 0, binding = 0, std430) buffer LumaBuf { float luma[]; };   // luma pyramids of a and c
layout(set = 0, binding = 1, std430) buffer FieldBuf { vec2 field[]; };  // half-vector fields
layout(set = 0, binding = 2, std430) buffer UBuf { uint u[]; };          // counters, medians, packed mismatch
layout(set = 0, binding = 3) uniform sampler2D texPrev;                  // frame a (linear, clamp to edge)
layout(set = 0, binding = 4) uniform sampler2D texCurr;                  // frame c
layout(set = 0, binding = 5, rgba8) uniform writeonly image2D outImg;

// One push-constant block for every pass; each pass documents which fields it reads. 128 bytes (full).
layout(push_constant) uniform PC {
    int w, h;        // level size (unpadded)
    int bw, bh;      // block grid of this level (synth passes: the finest level)
    int pw, ph;      // parent block grid / source size
    int offA, offC;  // luma offsets of this level
    int srcA, srcC;  // luma offsets of the source level
    int fIn, fOut;   // field offsets
    int fPar;        // parent field offset
    int radius;
    int medIdx;      // index into u[] of this level's global median (-1 = coarsest level)
    int W, H;        // full resolution
    int fw, fh;      // flow resolution
    int pf;          // pixel-field offset
    int fscale;      // FLOW_SCALE: flow res = full res / fscale (1, 2, 4)
    int sadAcc;      // search: also add the winning SAD to the scene-cut sum (finest level, sub-pixel off)
    // per generated frame (synth passes), t = time of the generated frame in (0, 1)
    float s;         // lookup: 2t - 1
    float wA, wC;    // blend weights 1 - t, t; warp a(q - 2 wC g), c(q + 2 wA g); wC == 0.5: the exact 2x path
    // runtime settings (framegen::Settings)
    float lambda, smoothBonus, zeroBonus, cutSad, trust0, trustScale, globalFrac;
} pc;

const int BLOCK = 8;
#define LAMBDA pc.lambda
#define CUT_SAD pc.cutSad
#define SMOOTH_BONUS pc.smoothBonus
#define ZERO_BONUS pc.zeroBonus
#define TRUST0 pc.trust0
#define TRUST_SCALE pc.trustScale   // 1 / (TRUST1 - TRUST0): the folded constant of a constant division
#define GLOBAL_FRAC pc.globalFrac
#define NEAR_C (pc.wC > 0.5)   // fallback frame: a for t <= 0.5, c after
#define MID (pc.wC == 0.5)     // t == 0.5: blend 0.5 * (wa + wc) exactly as the 2x path
const int E_OFF = 64;          // packed mismatch in u[], after the counters and medians
#define E_STRIDE ((pc.W + 3) >> 2)
const int U_UNTRUSTED = 0, U_SADSUM = 1, U_MEDIAN = 4;

// luma texel with clamp-to-edge (== cv2 BORDER_REPLICATE, and == the replicate pad_to_block)
float L(int off, int x, int y, int w, int h) {
    return luma[off + clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)];
}

// cv2.remap INTER_LINEAR / BORDER_REPLICATE on a float image: map rounded to 1/32 px, taps clamped.
// cv2 coordinates: integer = pixel centre, so p is used as-is (no +0.5).
float sampleL(int off, vec2 p, int w, int h) {
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    float p00 = L(off, i.x, i.y, w, h), p01 = L(off, i.x + 1, i.y, w, h);
    float p10 = L(off, i.x, i.y + 1, w, h), p11 = L(off, i.x + 1, i.y + 1, w, h);
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x)
         + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}

// cv2.remap INTER_LINEAR / BORDER_REPLICATE on the 2-channel float field at offset off (fw x fh)
vec2 F(int off, int x, int y) {
    return field[off + clamp(y, 0, pc.fh - 1) * pc.fw + clamp(x, 0, pc.fw - 1)];
}
vec2 sampleF(int off, vec2 p) {
    ivec2 X = ivec2(roundEven(p * 32.0));
    ivec2 i = X >> 5;
    vec2 f = vec2(X & 31) * (1.0 / 32.0);
    vec2 p00 = F(off, i.x, i.y), p01 = F(off, i.x + 1, i.y);
    vec2 p10 = F(off, i.x, i.y + 1), p11 = F(off, i.x + 1, i.y + 1);
    return p00 * ((1.0 - f.y) * (1.0 - f.x)) + p01 * ((1.0 - f.y) * f.x)
         + p10 * (f.y * (1.0 - f.x)) + p11 * (f.y * f.x);
}

// full-res half-vector at pixel p: cv2.resize(pf, fscale x, INTER_LINEAR) (+ edge pad) * FLOW_SCALE
vec2 halfVec(ivec2 p) {
    int S = pc.fscale;
    p = min(p, ivec2(S * pc.fw - 1, S * pc.fh - 1));
    vec2 s = (vec2(p) + 0.5) * (1.0 / float(S)) - 0.5;
    vec2 fl = floor(s);
    vec2 f = s - fl;
    ivec2 i0 = clamp(ivec2(fl), ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    ivec2 i1 = clamp(ivec2(fl) + 1, ivec2(0), ivec2(pc.fw - 1, pc.fh - 1));
    vec2 v00 = field[pc.pf + i0.y * pc.fw + i0.x], v01 = field[pc.pf + i0.y * pc.fw + i1.x];
    vec2 v10 = field[pc.pf + i1.y * pc.fw + i0.x], v11 = field[pc.pf + i1.y * pc.fw + i1.x];
    return float(S) * mix(mix(v00, v01, f.x), mix(v10, v11, f.x), f.y);
}

// cv2.remap of a uint8 RGB frame at p (pixel-centre coords): bilinear, clamp to edge, rounded to 8 bit.
vec3 sampleRGB(sampler2D t, vec2 p) {
    p = roundEven(p * 32.0) * (1.0 / 32.0);
    return floor(textureLod(t, (p + 0.5) / vec2(pc.W, pc.H), 0.0).rgb * 255.0 + 0.5);
}

uint mismatch(ivec2 p, out vec3 wa, out vec3 wc) {
    vec2 hv = halfVec(p);
    // a(q - 2t g), c(q + 2(1 - t) g); float32 product then sum like numpy (2t = 2(1 - t) = 1 at t = 0.5: exact)
    precise float ka = 2.0 * pc.wC, kc = 2.0 * pc.wA;   // exact: interp's float32(2t), float32(2(1 - t))
    precise vec2 pa = vec2(p) - ka * hv;
    precise vec2 pc_ = vec2(p) + kc * hv;
    wa = sampleRGB(texPrev, pa);
    wc = sampleRGB(texCurr, pc_);
    vec3 d = abs(wa - wc);
    return uint(max(d.r, max(d.g, d.b)));
}
