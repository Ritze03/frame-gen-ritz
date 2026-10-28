// MotionVersion::Learned: the learned motion estimator (ml-fg's export/SPEC.md, the porting contract, and export/reference.py, its
// executable reference; nets e2v3 / e1v3). It replaces V1's motion search only: P_LUMA's flow-res luma in, V1's per-pixel field
// (field[pc.pf], as pixfield.comp writes it) and scene-cut sum (u[U_SADSUM], as search.comp / subpix.comp add it) out; V1's
// synthesis reads them unchanged. Included after common.glsl (the bindings every pass shares). framegen.cpp MLPC mirrors this block;
// it is pushed over the same 128-byte range of the same pipeline layout as V1's PC.
//
// Net (SPEC.md section 2): the flow-res lumas padded (edge replication) to a multiple of 16 per axis, / 255, a 5-level 2x2-mean
// pyramid; per level, coarse to fine: the coarser field upsampled x2 (bilinear, half-pixel centres) and doubled, a 14-channel input
// (both lumas warped by -h / +h, their difference, a 3x3 cost volume, h / 4), four 3x3 convs with clamp(0, 1) and a linear 3x3 conv
// to 2 channels, h += 2 * out. Everything is float32 (SPEC.md section 7: the field h and its updates must stay float32).

layout(push_constant) uniform MLPC {
    int W, H;          // full resolution
    int fw, fh;        // flow resolution (FGR's level 0: the luma of P_LUMA, the field the synthesis reads)
    int pw, ph;        // the net's level 0: fw, fh padded to a multiple of 16 (level l is (pw >> l) x (ph >> l))
    int lvl;           // the level of this pass (0 = finest, 4 = coarsest)
    int actIn, actOut; // conv / match: vec4 offsets in act[] of the input / output activation (channel groups of 4, planar)
    int wOff, bOff;    // conv: the layer's weights / bias in wt[] (vec4 units, gpu/ml/embed_weights.py's layout)
    int nPart;         // reduce: number of partial sums
    int offYa, offYc;  // luma[] offsets of the flow-res lumas of a and c (0..255)
    int bw, bh;        // the finest block grid of V1 (fallback.comp's scene-cut mean divides by bw * bh)
    int pf;            // field[] offset of the per-pixel half-vector field (V1's fieldPF)
    int mode;          // reduce: 0 = the duplicate test, 1 = the scene-cut sum
    float dupDiff;     // reduce, mode 0: the duplicate threshold (mean |a - c| per channel, levels)
    int ext;           // the net variant's extra per-frame planes beyond the luma (framegen.cpp kMlVariants: 0 luma, 2 colour = Cb, Cr)
} pc;

// Binding 13: the net's float state, laid out per the padded level sizes n(l) = (pw >> l) * (ph >> l), S = n(0) + .. + n(4):
//   [0, 2S)    the pyramid: per level, la then lc (0..1)
//   [2S, 4S)   the field h per level (x, y interleaved, in that level's pixels)
//   [4S, 4S + 2 ext S)  the variant's extra planes' pyramids (ext = pc.ext, 0 for a luma net), per level: for each extra plane p
//              (colour: 0 = Cb, 1 = Cr) a then c (/255, signed); the planes are those of ml_chroma.comp (the variant's per-frame pass)
//   [.., + 2 ext fw fh)  ml_chroma.comp's output, the flow-res extra planes (levels), plane p of frame f (0 = a, 1 = c) at (2 p + f) fw fh
//   [.., ..)   partial sums of the reductions (ml_dup.comp, ml_out.comp -> ml_reduce.comp)
// Binding 14: the activations, vec4 (4 channels), two regions of level-0 size: R0 (the 14-channel input (25 for colour), then layers 1 and 3) and
// R1 (layers 0 and 2), each channel group planar per level. Binding 15: the weights (both nets, gpu/ml/embed_weights.py's layout).
// All three are bound only to the sets of MotionVersion::Learned (framegen.cpp MotionGroup::bufs / mlWeights).
layout(set = 0, binding = 13, std430) buffer MlState { float ml[]; };
layout(set = 0, binding = 14, std430) buffer MlAct { vec4 act[]; };
layout(set = 0, binding = 14, std430) buffer MlActH { uvec2 acth[]; };   // the same buffer, ML_HALF's layout
layout(set = 0, binding = 15, std430) restrict readonly buffer MlWeights { vec4 wt[]; };

// Settings::learnedHalf (a specialization constant of ml_match / ml_conv): the activations (the 14-channel input and the four hidden
// layers' outputs) are stored as IEEE half, 4 channels in a uvec2, arithmetic stays float32 (SPEC.md section 7, the tested fp16
// storage; the field h, the pyramid and the weights stay float32)
layout(constant_id = 24) const int ML_HALF = 0;
// Round to nearest even like numpy's astype(float16) (what SPEC.md measured): RADV's packHalf2x16 truncates (v_cvt_pkrtz), so the
// float32 mantissa is first rounded to half's 10 bits (ties to even) in the integer domain, which the truncation then keeps exactly.
// (Values below half's normal range, |x| < 6.1e-5, are still truncated to its subnormal grid.)
vec4 mlRoundHalf(vec4 v) {
    uvec4 u = floatBitsToUint(v);
    u = (u + 0xFFFu + ((u >> 13) & 1u)) & ~0x1FFFu;
    return uintBitsToFloat(u);
}
uvec2 mlPack(vec4 v) { v = mlRoundHalf(v); return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw)); }
vec4 mlUnpack(uvec2 p) { return vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y)); }
vec4 actLoad(int i) { return ML_HALF != 0 ? mlUnpack(acth[i]) : act[i]; }
void actStore(int i, vec4 v) {
    if (ML_HALF != 0) acth[i] = mlPack(v);
    else act[i] = v;
}

const int ML_LEVELS = 5;
const int U_MLDUP = 2;   // u[2] = 1: the pair is a duplicate (ml_reduce.comp, mode 0): every pass of the net exits at once,
                         // ml_out.comp writes a zero field and the scene-cut sum stays 0 (FGR's exact answer for a still pair)

bool mlDup() { return u[U_MLDUP] != 0u; }
int mlN(int l) { return (pc.pw >> l) * (pc.ph >> l); }
int mlS() { int s = 0; for (int l = 0; l < ML_LEVELS; l++) s += mlN(l); return s; }
int mlPyr(int l) { int o = 0; for (int k = 0; k < l; k++) o += 2 * mlN(k); return o; }   // la of level l; lc at + mlN(l)
int mlH(int l) { int o = 2 * mlS(); for (int k = 0; k < l; k++) o += 2 * mlN(k); return o; }
int mlX(int l) { int o = 4 * mlS(); for (int k = 0; k < l; k++) o += 2 * pc.ext * mlN(k); return o; }   // extra plane pyramids of level l
int mlFlowX() { return 4 * mlS() + 2 * pc.ext * mlS(); }                                              // ml_chroma.comp's flow-res planes
int mlPart() { return mlFlowX() + 2 * pc.ext * pc.fw * pc.fh; }
