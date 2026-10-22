// synth/error: full-res pass. Upsample the per-pixel flow field (fgHalfVec), warp prev and curr (RGB, 1/32-quantised
// bilinear, rounded to 8-bit levels like cv2.remap on uint8), e = max-channel |wa - wc| (integer-valued).
// error.comp:9-20, common.glsl:130-142; interp.py:229-236.
// MRT: att0 = mismatch (R8, e/255), att1 = warpA, att2 = warpC (RGBA8 levels / 255; RgbaF16: RGBA16F values).
uniform sampler2D uPrev;    // frame[prevS] (RGBA8)
uniform sampler2D uCurr;    // frame[currS] (RGBA8)
uniform sampler2D uField;   // pixfield if t == 0.5 else fieldT (RG32F, fw x fh)
uniform int uFlowScale;
uniform float uWA;          // fround(1-t)
uniform float uWC;          // fround(t)

layout(location = 0) out vec4 oE;
layout(location = 1) out vec4 oWarpA;
layout(location = 2) out vec4 oWarpC;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 hv = fgHalfVec(uField, p, uFlowScale);
    // a(q - 2t g), c(q + 2(1-t) g); 2t = 2*uWC and 2(1-t) = 2*uWA (exact: power-of-two scale)
    float ka = 2.0 * uWC;
    float kc = 2.0 * uWA;
    vec3 wa = fgTap(uPrev, vec2(p) - ka * hv);   // Rgba8: integer-valued 0..255; RgbaF16: the source's values
    vec3 wc = fgTap(uCurr, vec2(p) + kc * hv);
    vec3 d = FG_D(wa, wc);
    float e = floor(min(max(d.r, max(d.g, d.b)), 255.0));   // integer 0..255 (FG_PACKE)
    oE = vec4(e / 255.0, 0.0, 0.0, 1.0);
    oWarpA = fgPackW(wa);
    oWarpC = fgPackW(wc);
}
