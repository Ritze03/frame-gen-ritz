// V2 motion blur (engine.synthBlur, the library's recordSynthBlur with MotionVersion::V2: v2_morph.glsl pixelOut with n samples): one
// sub-frame (the stored output of synth(t_k): the 8-bit / half-float sub-frame, or a real frame) times its weight, added to the
// RGBA32F accumulator by the host's ONE, ONE blend (the first sample is written without blending). Working values (levels for RGBA8).
uniform sampler2D uSrc;
uniform float uWeight;
layout(location = 0) out vec4 oAcc;

void main() { oAcc = vec4(uWeight * fgPx(uSrc, ivec2(gl_FragCoord.xy)), 0.0); }
