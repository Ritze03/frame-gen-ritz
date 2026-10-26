// Zero-motion test, step 2 (SPEC stage 7 steps 3-5, reference.py zero_test / box; settings.learnedZeroTest): per flow pixel q of the
// fw x fh grid, the 5x5 box MEANS of learned/zt_cost.frag.glsl's d and e with replicate border (a tap outside the frame reads the
// cost of the nearest pixel: cv2.blur BORDER_REPLICATE), summed like reference.py box(): each row's 5 taps left to right, then the 5
// row sums top to bottom, then / 25:
//   C(h) = box5(d), C(0) = box5(e);  zero = (C(0) - ML_ZERO_BONUS) <= C(h)  ("<=": zero wins ties);  pf' = zero ? 0 : h
// MRT: att0 = pixfield (the final field, RG32F), att1 (ml.zt, RGBA32F): r = the SAD term of q under the final field (zero ? e : d,
// SPEC stage 7 "the SAD comes for free": the input of the sadRed chain, so sadSum belongs to pf'), g = C(h), b = C(0), a = zero.
// A duplicate pair (mlDup) writes 0 (its net field is 0 too, so the result would be the same).
const int ML_ZERO_WIN = 5;          // reference.py ZERO_WIN (FGR's PIX_WIN): do not change, the net is scored with it
const float ML_ZERO_BONUS = 1.0;    // reference.py ZERO_BONUS, luma levels
uniform sampler2D uCost;    // ml.ztCost: r = d, g = e (fw x fh)
uniform sampler2D uField;   // ml.pfNet: the net's field (fw x fh)
layout(location = 0) out vec4 oField;
layout(location = 1) out vec4 oZt;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    if (mlDup()) { oField = vec4(0.0); oZt = vec4(0.0); return; }
    ivec2 fs = textureSize(uCost, 0);
    const int r = ML_ZERO_WIN / 2;
    vec2 sum = vec2(0.0);
    for (int j = -r; j <= r; j++) {
        int y = clamp(q.y + j, 0, fs.y - 1);
        vec2 row = vec2(0.0);
        for (int i = -r; i <= r; i++) row += texelFetch(uCost, ivec2(clamp(q.x + i, 0, fs.x - 1), y), 0).rg;
        sum += row;
    }
    vec2 c = sum / float(ML_ZERO_WIN * ML_ZERO_WIN);   // x = C(h), y = C(0)
    bool zero = (c.y - ML_ZERO_BONUS) <= c.x;
    vec2 own = texelFetch(uCost, q, 0).rg;
    oField = vec4(zero ? vec2(0.0) : texelFetch(uField, q, 0).xy, 0.0, 0.0);
    oZt = vec4(zero ? own.y : own.x, c.x, c.y, zero ? 1.0 : 0.0);
}
