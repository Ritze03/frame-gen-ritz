// V3.5: the camera model of v3_stats.comp (G_AFF) as a vote for a block. Included after v3_common.glsl, v3_est.glsl.

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


// round 2: the regional models (v35_rgn.comp: 4 x 3 regions, words G35_REG + 8 r) blended bilinearly between the region centres: a piecewise-affine camera. A region without
// its own fit (flag 2) predicts with the global model, flag 0 (no model at all) does not vote. Returns false when no region has a model.
const int G35_REG = 3900;
bool regPredict(vec2 c, out vec2 d, out float q) {
    d = vec2(0.0);
    q = 0.0;
    const vec2 cc = vec2(pc.cx, pc.cy);
    const vec2 u = c / (2.0 * cc) * vec2(4.0, 3.0) - 0.5;
    const ivec2 i0 = ivec2(floor(u));
    const vec2 f = u - vec2(i0);
    const vec3 p = vec3(1.0, (c - cc) / cc);
    float ws = 0.0;
    for (int k = 0; k < 4; k++) {
        const ivec2 r = clamp(i0 + ivec2(k & 1, k >> 1), ivec2(0), ivec2(3, 2));
        const float wt = ((k & 1) == 1 ? f.x : 1.0 - f.x) * ((k >> 1) == 1 ? f.y : 1.0 - f.y);
        const int o = gOff() + G35_REG + 8 * (r.y * 4 + r.x);
        if (ax[o + 7] == 0u || wt <= 0.0) continue;
        const vec3 a = vec3(uintBitsToFloat(ax[o]), uintBitsToFloat(ax[o + 1]), uintBitsToFloat(ax[o + 2]));
        const vec3 b = vec3(uintBitsToFloat(ax[o + 3]), uintBitsToFloat(ax[o + 4]), uintBitsToFloat(ax[o + 5]));
        d += wt * vec2(dot(a, p), dot(b, p));
        q += wt * uintBitsToFloat(ax[o + 6]);
        ws += wt;
    }
    if (ws <= 0.0) return false;
    d /= ws;
    q /= ws;
    return true;
}
// the regional camera at block b's centre as this level's vector of direction dir (level px)
bool regVote(const V3Lv L, int l, int dir, ivec2 b, out vec2 v, out float q) {
    v = vec2(0.0);
    q = 0.0;
    const float sc = float(1 << l);
    const vec2 c = l == 0 ? vec2(b * 8 + 4) : vec2(b * 4 + 2) * sc;
    vec2 d;
    if (!regPredict(c, d, q)) return false;
    if (dir == 1) {
        if (!regPredict(c - d, d, q)) return false;
        d = -d;
    }
    v = d / sc;
    return true;
}
