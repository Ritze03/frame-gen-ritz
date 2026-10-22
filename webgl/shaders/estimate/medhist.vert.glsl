#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
// Scatter pass of the global-median histogram (medhist.comp:24-30, LAYOUT 3.5). One point per (fine block,
// component): gl_VertexID = 2*blk + comp. Lands on texel (bin, comp) of the 256x2 hist target.
uniform sampler2D uParent;   // field[i+1]
uniform ivec2 uBlocks;       // fine level block grid (bw, bh)

void main() {
    int blk = gl_VertexID >> 1;
    int comp = gl_VertexID & 1;
    int ix = blk % uBlocks.x, iy = blk / uBlocks.x;
    ivec2 pb = textureSize(uParent, 0);
    int px = min(ix / 2, pb.x - 1), py = min(iy / 2, pb.y - 1);
    int v = int(roundEven(texelFetch(uParent, ivec2(px, py), 0)[comp]));
    int bin = clamp(v + 128, 0, 255);
    gl_Position = vec4((float(bin) + 0.5) / 256.0 * 2.0 - 1.0, (float(comp) + 0.5) / 2.0 * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = 1.0;
}
