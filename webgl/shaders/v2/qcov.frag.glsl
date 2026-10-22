// V2 Quality, coverage step 2b (flow resolution; gpu/shaders/v2_qcov.comp): qvis box-filtered over (2 Q_R + 1)^2 locations
// (replicate clamp) into the coverage (cP, cC), quantised to 11 bits each and the static-overlay mask z to 10 bits exactly as the
// library packs its coverage word (v2_quality.glsl qPackCov / qUnpackCov). Output v2.qcov: x = cP, y = cC, z = z.
uniform sampler2D uVis;   // v2.qvis
uniform sampler2D uQf;    // v2.qf: z
layout(location = 0) out vec4 oCov;

void main() {
    ivec2 m = ivec2(gl_FragCoord.xy);
    ivec2 mx = textureSize(uVis, 0) - 1;
    vec2 s = vec2(0.0);
    for (int j = -Q_R; j <= Q_R; j++)
        for (int i = -Q_R; i <= Q_R; i++) s += texelFetch(uVis, clamp(m + ivec2(i, j), ivec2(0), mx), 0).xy;
    vec2 c = s * (1.0 / float((2 * Q_R + 1) * (2 * Q_R + 1)));
    float z = texelFetch(uQf, m, 0).z;
    vec2 cq = roundEven(clamp(c, 0.0, 1.0) * 2047.0) * (1.0 / 2047.0);
    float zq = roundEven(clamp(z, 0.0, 1.0) * 1023.0) * (1.0 / 1023.0);
    oCov = vec4(cq, zq, 0.0);
}
