// V2 motion blur, the output: the weighted sum of the sub-frames / the sum of the weights, stored as the output format stores it
// (RGBA8: rounded levels; RGBA16F: as is). v2_morph.glsl pixelOut: imageStore(outImg, p, FG_STORE(acc / ws, ...)).
uniform sampler2D uAcc;
uniform float uWsum;
layout(location = 0) out vec4 oColor;

void main() { oColor = fgStore(texelFetch(uAcc, ivec2(gl_FragCoord.xy), 0).rgb / uWsum); }
