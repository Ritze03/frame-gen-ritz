// The net's SAD in V1's scene-cut contract (SPEC 5.3): sadSum (T.sadRed's last texel, 1x1) = sad * bw * bh, so that V1's
// meanSad = sadSum / (bw * bh) (synth/final.frag.glsl, stats()) is the net's sad = (sum over the fw x fh flow pixels) / (fw * fh).
// uWeight = bw * bh / (fw * fh). A duplicate pair (mlDup) gives 0.
uniform sampler2D uSrc;   // 1x1: the sum of learned/sad.frag.glsl's terms
uniform float uWeight;
layout(location = 0) out vec4 oSum;

void main() {
    oSum = vec4(mlDup() ? 0.0 : texelFetch(uSrc, ivec2(0), 0).r * uWeight);
}
