// V2 Performance synthesis, step 0 (full resolution): the pixel's own taps' mismatch em (v2_morph.glsl morph(), up to the block-vector
// candidates). Read by v2/group (4x4 group sums, the candidate gate) and by v2/morph (partial groups at the frame edge).
layout(location = 0) out vec4 oEm;

void main() {
    oEm = vec4(front(ivec2(gl_FragCoord.xy), uT).em, 0.0, 0.0, 0.0);
}
