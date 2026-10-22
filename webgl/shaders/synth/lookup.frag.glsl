// synth/lookup: field_at(pf, t) (interp.py:194-205, lookup.comp:12-17). Content at midpoint p sits at
// q = p + (2t-1) h(p) at time t, so g(q) = h(q - (2t-1) g(q)): LOOKUP_ITERS = 2 bilinear fixed-point steps
// started from h(q). Not run when t == 0.5 (uS == 0).
uniform sampler2D uField;   // pixfield (RG32F, fw x fh)
uniform float uS;           // fround(2t-1)

layout(location = 0) out vec4 oField;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    vec2 g = texelFetch(uField, q, 0).xy;
    for (int k = 0; k < 2; k++) {   // LOOKUP_ITERS
        g = fgSampleF(uField, vec2(q) - uS * g);   // float32 product then difference, like numpy
    }
    oField = vec4(g, 0.0, 0.0);
}
