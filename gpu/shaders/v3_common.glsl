// MotionVersion::V3 (design: gpu/V3.md): the push-constant block, the three V3 storage buffers (bindings 13, 14, 15) and the layout
// helpers every V3 pass shares. Included after common.glsl (the bindings 0-5 and 11/12 of every pass, the format). Do not include
// motion_v1.glsl or v2_motion.glsl in the same shader (each declares a push-constant block and V2's binding 13). Section numbers below are
// those of gpu/V3.md.

// Section 5. Pushed over the same 128-byte range of the same pipeline layout as PC / V2PC (framegen.cpp V3PC, static_assert on the size).
// The first 8 ints are V2PC's, so v2_static.comp can be pushed with this block.
layout(push_constant) uniform V3PC {
    int W, H, fw, fh, bw, bh, fscale, fIn;   // full / flow resolution; bw, bh = level-0 block grid (8x8 flow px); fIn = 1: Settings::v2StaticMask off
    int lw, lh, gw, gh;                      // this pass's level size and block grid (synth: unused)
    int lvl, iter;                           // level, iteration (synth: lvl = interior sample count, iter = first interior sample index; unused now)
    int mode, n;                             // mode bits (per pass, below); n = samples of this output (1 = recordSynth)
    float t0, t1;                            // sample window; recordSynth: t0 = t1 = t
    float cutThresh, globalFrac;             // Settings::v3SceneCut, Settings::globalFallback
    float smear, occlSoft, unsure;           // Settings::v3Smear, v3OcclSoft, v3Unsure
    float cx, cy;                            // model centre x, y (fw/2, fh/2) (stats), spare otherwise
    int fdOff;                               // Y1: the word offset of FD in V3B_FLOW (= 16 B: after the block records); FF = fdOff + 4 N, TF = fdOff + 8 N
    float hudB;                              // task 105: Settings::hudBonus in V3 cost units (0 = off); v3_est.glsl
    float hp[5];                             // task 105: the ramps of the bonus (v3_est.glsl)
} pc;
// mode bits, per pass (a pass never reads another's meaning): search: bit 0 = hook (j) colour window; dense: bit 0 = (j), bit 2 = inject
// (tests); synth: bit 0 = Gaussian blur weights, bit 1 = hook (l) detail split, bits 4..7 = Settings::v3Blend's variant 1..V3B_NETS (1 Fluidity, 2 Consistency) (the -DV3B=1 build only), bit 3 = debug dump (then lw = the stride of the dumped pixels in the
// raster, lh = their number).
const int V3M_GAUSS = 1, V3M_DETAIL = 2, V3M_DUMP = 8;
// (Y1) occl: bit 2 = inject (tests, BLK is stale: amb = 0); fill, project: none.

// Binding 13: V3B_PYR, raw planes and features of both frames, all levels (uvec2 per texel, section 4); binding 14: V3B_AUX, static
// layout + maps + the pair's state G (uint words); binding 15: V3B_FLOW, block records, dense and filled fields, the T-field (floats).
#ifndef V3_RO
#define V3_RO
#endif
layout(set = 0, binding = 13, std430) V3_RO buffer V3Pyr { uvec2 pyr[]; };
layout(set = 0, binding = 14, std430) V3_RO buffer V3Aux { uint ax[]; };
layout(set = 0, binding = 15, std430) V3_RO buffer V3Flow { float fl[]; };

// V3B_AUX word offsets (section 4; N = fw * fh, N8 = the 8x8 cells of the flow grid). sOff .. cOff are exactly v2_motion.glsl's (they must
// change together): v2_static.comp writes S and the cell maxima, v3_enclose.comp E and C.
int s8W() { return (pc.fw + 7) >> 3; }
int s8H() { return (pc.fh + 7) >> 3; }
int mapOff(int d) { return d * pc.fw * pc.fh; }        // MAP[d]: cost8 | e8 << 8 | amb8 << 16 | g8 << 24
int sOff() { return 4 * pc.fw * pc.fh; }
int s8Off() { return 5 * pc.fw * pc.fh; }
int eOff() { return s8Off() + s8W() * s8H(); }
int cOff() { return eOff() + s8W() * s8H(); }
int gOff() { return cOff() + s8W() * s8H(); }
// the state G (word index from gOff(), 4096 words, zeroed by every estimate)
const int G_SUM = 0, G_UNSURE_COUNT = 2, G_COST_SUM = 3, G_MODE = 4, G_UNSURE_COPY = 5, G_VALID = 6, G_DOM = 8, G_AFF = 16, G_CAM_TRANS = 22,
          G_FAIL_MIX = 24, G_TK = 1200 /* the sample times of a synth record (floats, up to 16: written by the host) */, G_WIDE_FLAGGED = 25, G_WIDE_REPLACED = 26, G_SUMSQ = 27, G_HIST_D = 29 /* v3_decide: the scene-cut divergence D (float bits) */,
          G_ELIG = 30 /* v3_occl: the pixels that could have found a partner (their vector and the camera model's both stay in the frame) */,
          G_UNSURE_ELIG = 31 /* ... of which unsure (u >= 0.5): v3_decide's share */, G_HIST = 32;
