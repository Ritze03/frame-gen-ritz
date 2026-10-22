// V2 final: the morphed sub-frame, or - only for a whole-frame decision - the nearer real frame (prev for t <= 0.5, curr after): a scene cut
// (mean finest SAD of the re-picked block vectors > sceneCutSad) or a flick (more than globalFallback of the frame in untrusted tiles).
// There is no local hold. v2_morph.glsl pixelOut(), v2_morph.comp fix-up pass.
uniform sampler2D uPrev;        // RGBA8
uniform sampler2D uCurr;
uniform sampler2D uMorph;       // RGBA8 sub-frame (v2/morph)
uniform sampler2D uSadSum;      // 1x1 R32F: sum of the blocks' roundEven(256 mean SAD) / 256
uniform sampler2D uUntrusted;   // 1x1 R32F
uniform ivec2 uFrameSize;
uniform int uNumBlocks;
uniform float uCutSad;
uniform float uGlobalFrac;
uniform int uNearC;
layout(location = 0) out vec4 oColor;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float meanSad = texelFetch(uSadSum, ivec2(0), 0).r / float(uNumBlocks);
    bool cut = meanSad > uCutSad;
    bool flick = texelFetch(uUntrusted, ivec2(0), 0).r > uGlobalFrac * float(uFrameSize.x) * float(uFrameSize.y);
    if (cut || flick) oColor = vec4((uNearC != 0 ? texelFetch(uCurr, p, 0) : texelFetch(uPrev, p, 0)).rgb, 1.0);
    else oColor = vec4(texelFetch(uMorph, p, 0).rgb, 1.0);
}
