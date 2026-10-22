// synth/hbox: horizontal 7-sum of the integer mismatch e (replicate border). resolve.comp:27-31; interp.py:236.
// hsum is R16UI: max 7*255 = 1785.
uniform sampler2D uE;       // mismatch (R8, e/255)
uniform ivec2 uFrameSize;

layout(location = 0) out uvec4 oH;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    uint s = 0u;
    for (int k = -3; k <= 3; k++) {
        int x = clamp(p.x + k, 0, uFrameSize.x - 1);
        s += uint(floor(texelFetch(uE, ivec2(x, p.y), 0).r * 255.0 + 0.5));
    }
    oH = uvec4(s, 0u, 0u, 0u);
}
