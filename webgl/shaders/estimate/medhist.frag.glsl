// Global-median histogram of the parent vectors as a gather (medhist.comp:24-30), cumulative per band: each fine block counts its
// parent's vector once, so a parent texel weighs as many fine blocks as map to it (min(ix / 2, pw - 1)): 2 x 2 inside, fewer at the
// right / bottom edge. Target hist[i]: 64 x (2 B) RGBA32F, texel (g, comp * B + band) component j = the weight of the parents of rows
// [MH_ROWS band, MH_ROWS (band + 1)) whose component comp rounds (roundEven) to at most bin - 128 (bin = 4 g + j, clamped to 0..255).
// medsel sums the B bands. Integer sums: exact in fp32.
const int MH_ROWS = 4;   // parent rows per band (= the host's; more bands = more fragments, the scan is latency-bound)
uniform sampler2D uParent;   // field[i + 1]
uniform ivec2 uBlocks;       // fine level block grid (bw, bh)
layout(location = 0) out vec4 oCount;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    ivec2 pb = textureSize(uParent, 0);
    int B = (pb.y + MH_ROWS - 1) / MH_ROWS;
    int comp = o.y / B, band = o.y - comp * B;
    vec4 bins = vec4(4 * o.x) + vec4(0.0, 1.0, 2.0, 3.0);
    vec4 n = vec4(0.0);
    for (int py = MH_ROWS * band; py < min(MH_ROWS * band + MH_ROWS, pb.y); py++) {
        int cy = py < pb.y - 1 ? clamp(uBlocks.y - 2 * py, 0, 2) : max(uBlocks.y - 2 * py, 0);
        for (int px = 0; px < pb.x; px++) {
            int cx = px < pb.x - 1 ? clamp(uBlocks.x - 2 * px, 0, 2) : max(uBlocks.x - 2 * px, 0);
            float bin = float(clamp(int(roundEven(texelFetch(uParent, ivec2(px, py), 0)[comp])) + 128, 0, 255));
            n += float(cx * cy) * vec4(greaterThanEqual(bins, vec4(bin)));
        }
    }
    oCount = n;
}
