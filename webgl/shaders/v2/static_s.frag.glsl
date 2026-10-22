// V2 static-structure mask, step 2 (v2_static.comp, flow resolution): S = raw, dropped where fewer than 2 of the 8 neighbours agree as well
// (raw >= 0.5; a lone agreeing pixel is a moving texture that happens to match, the strokes of text and edges of HUD elements are lines).
// Outside the frame raw is 0. uStaticOff != 0 (settings.v2StaticMask off) writes an empty mask.
uniform sampler2D uRaw;
uniform int uStaticOff;
layout(location = 0) out vec4 oS;

void main() {
    ivec2 m = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(uRaw, 0);
    float s = 0.0;
    if (uStaticOff == 0) {
        float n = 0.0;
        for (int j = -1; j <= 1; j++)
            for (int i = -1; i <= 1; i++) {
                ivec2 c = m + ivec2(i, j);
                if ((i != 0 || j != 0) && all(greaterThanEqual(c, ivec2(0))) && all(lessThan(c, sz)))
                    n += texelFetch(uRaw, c, 0).r >= 0.5 ? 1.0 : 0.0;
            }
        s = texelFetch(uRaw, m, 0).r * clamp(n - 1.0, 0.0, 1.0);
    }
    oS = vec4(s, 0.0, 0.0, 0.0);
}
