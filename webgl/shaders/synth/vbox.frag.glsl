// synth/vbox: vertical 7-sum of hsum -> 7x7 box mean of e -> trust ramp. resolve.comp:38-42; interp.py:236-237.
// (The "t > 0.5" untrusted flag is taken in reduce_untrusted's first step, not stored.)
uniform usampler2D uH;      // hsum (R16UI)
uniform ivec2 uFrameSize;
uniform float uTrust0;
uniform float uTrustScale;  // fround(1/(trustHigh-trustLow)), folded on the CPU

layout(location = 0) out vec4 oTrust;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    uint s = 0u;                                  // exact integer sum
    for (int k = -3; k <= 3; k++) {
        int y = clamp(p.y + k, 0, uFrameSize.y - 1);
        s += texelFetch(uH, ivec2(p.x, y), 0).r;
    }
    float e = float(s) * (1.0 / 49.0);
    float t = clamp((e - uTrust0) * uTrustScale, 0.0, 1.0);
    oTrust = vec4(t, 0.0, 0.0, 1.0);
}
