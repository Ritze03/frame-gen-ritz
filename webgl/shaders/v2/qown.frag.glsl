// V2 Quality, the coverage splat's fragment (see v2/qown.vert.glsl): the claim word as the float with its bits, MIN-blended.
flat in uint vW;
layout(location = 0) out vec4 oClaim;

void main() { oClaim = vec4(uintBitsToFloat(vW), 0.0, 0.0, 0.0); }
