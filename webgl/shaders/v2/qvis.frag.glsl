// V2 Quality, coverage step 2a (flow resolution; the per-location part of gpu/shaders/v2_qcov.comp): does prev / curr still show the
// content of location m: it won its footprint there, or nearly (its own key within Q_MARGIN levels of the winner's), or the winner
// moves the same way (within Q_TOL), or the winner matches badly itself (worse than Q_WIN levels); a footprint outside the frame: no.
// Output v2.qvis: x = prev, y = curr (0 / 1).
uniform highp isampler2D uClaim;   // v2.qclaim
uniform sampler2D uKeyA;           // v2.qkeyA / C: the winning key per footprint pixel
uniform sampler2D uKeyC;
uniform sampler2D uVecA;           // v2.qvecA / C: the winning vector bits
uniform sampler2D uVecC;
layout(location = 0) out vec4 oVis;

bool sees(float wk, float wvf, int own, vec2 hq) {
    return float(own) <= wk + Q_MARGIN * Q_KEY || wk > Q_WIN * Q_KEY || all(lessThanEqual(abs(qVec(int(wvf)) - hq), vec2(Q_TOL)));
}

void main() {
    ivec2 m = ivec2(gl_FragCoord.xy);
    ivec4 c = texelFetch(uClaim, m, 0);
    vec2 hq = qVec(c.y);   // the vector m splatted with (quantised)
    bool a = false, b = false;
    if (c.z >= 0) { ivec2 p = qUnpix(c.z); a = sees(texelFetch(uKeyA, p, 0).r, texelFetch(uVecA, p, 0).r, c.x, hq); }
    if (c.w >= 0) { ivec2 p = qUnpix(c.w); b = sees(texelFetch(uKeyC, p, 0).r, texelFetch(uVecC, p, 0).r, c.x, hq); }
    oVis = vec4(a ? 1.0 : 0.0, b ? 1.0 : 0.0, 0.0, 0.0);
}
