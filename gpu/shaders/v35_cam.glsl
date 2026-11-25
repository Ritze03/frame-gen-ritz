// V3.5: the camera model of v3_stats.comp (G_AFF) as a vote for a block, and its quality measure (written by v35_camq.comp). Included after v3_common.glsl, v3_est.glsl.
const int G35_CAMQ = 4000;   // word of the state G: the share of the confident blocks of the stats level that agree with the camera model (float bits)
float camQuality() { return smoothstep(0.55, 0.85, uintBitsToFloat(ax[gOff() + G35_CAMQ])); }

// the camera model (v3_stats.comp, G_AFF) at the block's centre, as this level's vector of direction dir; valid bit 8 of G_VALID
bool camVote(const V3Lv L, int l, int dir, ivec2 b, out vec2 v) {
    v = vec2(0.0);
    if ((ax[gOff() + G_VALID] & 256u) == 0u) return false;
    const int g = gOff() + G_AFF;
    const vec3 ca = vec3(uintBitsToFloat(ax[g]), uintBitsToFloat(ax[g + 1]), uintBitsToFloat(ax[g + 2]));
    const vec3 cb = vec3(uintBitsToFloat(ax[g + 3]), uintBitsToFloat(ax[g + 4]), uintBitsToFloat(ax[g + 5]));
    const float sc = float(1 << l);
    const vec2 c = l == 0 ? vec2(b * 8 + 4) : vec2(b * 4 + 2) * sc;
    const vec2 cc = vec2(pc.cx, pc.cy);
    vec2 d = vec2(dot(ca, vec3(1.0, (c - cc) / cc)), dot(cb, vec3(1.0, (c - cc) / cc)));
    if (dir == 1) {
        const vec2 q = c - d;
        d = -vec2(dot(ca, vec3(1.0, (q - cc) / cc)), dot(cb, vec3(1.0, (q - cc) / cc)));
    }
    v = d / sc;
    return true;
}

