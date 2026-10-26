// Zero-motion test, step 1 (SPEC stage 7 steps 1-2, reference.py zero_test; settings.learnedZeroTest): the two per-pixel costs of
// flow pixel q on the UNPADDED fw x fh grid, from the net's field (ml.pfNet, written by level 0's last conv in place of pixfield):
//   d(q) = |ya(q - h) - yc(q + h)|   h = the net's vector at q, true bilinear with clamp-to-edge on the fw x fh grid (mlBil)
//   e(q) = |ya(q) - yc(q)|           the cost of the zero vector
// ya, yc on the 0..255 scale (the net's level 0: b, a; the same planes the net got). learned/zt.frag.glsl box-filters both. The
// field cannot be overwritten in place (the 5x5 windows read the neighbours' net vectors), hence the separate input texture.
// out: r = d, g = e (RG32F). A duplicate pair (mlDup) writes 0.
uniform sampler2D uLv0;     // the net's level 0 (padded): b = ya, a = yc
uniform sampler2D uField;   // ml.pfNet: the net's field, fw x fh
layout(location = 0) out vec4 oCost;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    if (mlDup()) { oCost = vec4(0.0); return; }
    ivec2 fs = textureSize(uField, 0);
    vec2 h = texelFetch(uField, q, 0).xy;
    float d = abs(mlBil(uLv0, fs, vec2(q) - h).b - mlBil(uLv0, fs, vec2(q) + h).a);
    vec4 l = texelFetch(uLv0, q, 0);
    oCost = vec4(d, abs(l.b - l.a), 0.0, 0.0);
}
