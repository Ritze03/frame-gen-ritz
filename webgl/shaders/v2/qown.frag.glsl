// V2 Quality, the coverage splat's fragment (see v2/qown.vert.glsl): the point's key (pass 0) or vector bits (pass 1), MIN-blended.
flat in float vVal;
layout(location = 0) out vec4 oClaim;

void main() { oClaim = vec4(vVal, 0.0, 0.0, 0.0); }
