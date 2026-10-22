// V2 Quality, coverage step 2a (flow resolution; the per-location part of gpu/shaders/v2_qcov.comp): does prev / curr still show the
// content of location m: it won its footprint there, or nearly (its own key within Q_MARGIN levels of the winner's), or the winner
// moves the same way (within Q_TOL), or the winner matches badly itself (worse than Q_WIN levels); a footprint outside the frame: no.
// The winners are the claim words of v2/qown (float bits, see v2/qown.vert.glsl). Output v2.qvis: x = prev, y = curr (0 / 1).
uniform highp isampler2D uClaim;   // v2.qclaim
uniform sampler2D uOwnA;           // v2.qownA / C: the winning claim word per footprint pixel
uniform sampler2D uOwnC;
layout(location = 0) out vec4 oVis;

bool sees(float wf, int own, vec2 hq) {
    uint w = floatBitsToUint(wf) - 0x800000u;
    float wk = float(w >> 20);
    return float(own) <= wk + Q_MARGIN * Q_KEY || wk > Q_WIN * Q_KEY || all(lessThanEqual(abs(qVec(int(w & 0xFFFFFu)) - hq), vec2(Q_TOL)));
}

void main() {
    ivec2 m = ivec2(gl_FragCoord.xy);
    ivec4 c = texelFetch(uClaim, m, 0);
    vec2 hq = qVec(c.y);   // the vector m splatted with (quantised)
    bool a = c.z >= 0 && sees(texelFetch(uOwnA, qUnpix(c.z), 0).r, c.x, hq);
    bool b = c.w >= 0 && sees(texelFetch(uOwnC, qUnpix(c.w), 0).r, c.x, hq);
    oVis = vec4(a ? 1.0 : 0.0, b ? 1.0 : 0.0, 0.0, 0.0);
}
