// V2 flick count, step 1 (v2_morph.glsl pixelOut / main): an 8x8 tile is untrusted when the mean of its pixels' mismatch e exceeds
// trustMid (= (trustLow + trustHigh) / 2, V1's trust ramp midpoint); its valid (in-frame) pixels count. One fragment per tile; the
// 1x1 sum over the tiles (reduce_untrusted, uMode 0) is the library's u[cnt].
uniform sampler2D uE;   // R16F: e (integer levels)
uniform ivec2 uFrameSize;
uniform float uTrustMid;
layout(location = 0) out vec4 oTile;

void main() {
    ivec2 o = 8 * ivec2(gl_FragCoord.xy);
    ivec2 nv = min(ivec2(8), uFrameSize - o);
    float esum = 0.0;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            if (i < nv.x && j < nv.y) esum += texelFetch(uE, o + ivec2(i, j), 0).r;
    float valid = float(nv.x * nv.y);
    oTile = vec4(esum > uTrustMid * valid ? valid : 0.0, 0.0, 0.0, 0.0);
}
