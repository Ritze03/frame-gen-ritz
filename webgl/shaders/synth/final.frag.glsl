// synth/final: compose the output pixel. Global decisions (scene cut, untrusted share) are evaluated here from the
// 1x1 sadSum / untrusted textures and output the nearer real frame (prev for t <= 0.5, curr after).
// Otherwise out = mix(m, near, trust) with m = 0.5*(wa+wc) at t = 0.5 or wA*wa + wC*wc.
// fallback.comp:9-17; resolve.comp:72-81; interp.py:224-246 (TIME_BLEND).
// FG_SGUARD (the 'final_sg' variant, settings.learnedStaticGuard with the learned estimate; learned/sguard.inc.glsl): the generated
// pixel (as it would be stored) is then mixed towards the plain time mix uWA * prev + uWC * curr by wStatic; the fallbacks above and
// pixels with wStatic <= SG_MIN_W are left exactly as without it.
uniform sampler2D uPrev;        // frame[prevS] (RGBA8)
uniform sampler2D uCurr;        // frame[currS] (RGBA8)
uniform sampler2D uWarpA;       // RGBA8, integer levels / 255
uniform sampler2D uWarpC;
uniform sampler2D uTrust;       // R32F
uniform sampler2D uSadSum;      // 1x1 R32F
uniform sampler2D uUntrusted;   // 1x1 R32F
uniform ivec2 uFrameSize;
uniform int uNumBlocks;
uniform float uCutSad;
uniform float uGlobalFrac;
uniform float uWA;
uniform float uWC;
uniform int uNearC;
uniform int uMid;
#ifdef FG_SGUARD
uniform sampler2D uSg;          // learned/sg_v.frag.glsl: r = d, g = dMin (RG16F, full res)
uniform sampler2D uField;       // the field on the grid of t (pixfield at t = 0.5, else fieldT), flow res
uniform int uFlowScale;
#endif

layout(location = 0) out vec4 oColor;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 near = (uNearC != 0) ? texelFetch(uCurr, p, 0) : texelFetch(uPrev, p, 0);

    float meanSad = texelFetch(uSadSum, ivec2(0), 0).r / float(uNumBlocks);
    bool cut = meanSad > uCutSad;
    bool flick = texelFetch(uUntrusted, ivec2(0), 0).r > uGlobalFrac * float(uFrameSize.x) * float(uFrameSize.y);
    if (cut || flick) {
        oColor = vec4(near.rgb, 1.0);
        return;
    }

    vec3 wa = fgUnpackW(texelFetch(uWarpA, p, 0));
    vec3 wc = fgUnpackW(texelFetch(uWarpC, p, 0));
    float t = texelFetch(uTrust, p, 0).r;
    vec3 m = (uMid != 0) ? 0.5 * (wa + wc) : uWA * wa + uWC * wc;
    vec3 n8 = uNearC != 0 ? fgPx(uCurr, p) : fgPx(uPrev, p);
    vec3 o = m * (1.0 - t) + n8 * t;
#ifdef FG_SGUARD
#ifndef FG_F16
    o = clamp(floor(o + 0.5), 0.0, 255.0);   // the stored 8-bit output (fgStore's rounding; static_guard.py works on it)
#endif
    float w = sgWeight(texelFetch(uSg, p, 0).rg, uField, p, uFrameSize, 2.0 * max(uWA, uWC), uFlowScale);
    if (w > SG_MIN_W) o = o * (1.0 - w) + (uWA * fgPx(uPrev, p) + uWC * fgPx(uCurr, p)) * w;
#endif
    oColor = fgStore(o);
}
