// Motion version 2, the motion estimate (both V2Modes): helpers of v2_medhist / v2_search / v2_subpix. Included after common.glsl and
// motion_v1.glsl (V2's estimate pushes V1's PC block and computes V1's block field, faster).
//
// V1 runs a separate 3x3 median pass (median3.comp) after the search of every level; its output is read by the next finer level's
// search (4 parent candidates) and medhist, and by the sub-pixel pass. V2 drops those passes (4 dispatches and barriers, ~0.035 ms at
// 1080p) and takes the median where it is read, from the search's raw output. The median of the same 9 values is the same value, so
// the block field is V1's.

// The 3x3 per-component median (cv2.medianBlur ksize 3, replicate border) of the raw block field at offset off (grid gw x gh), at b.
// Both components at once: a min/max network (Devillard's opt_med9) on vec2.
#define V2_SORT(a, b) { vec2 lo_ = min(a, b); b = max(a, b); a = lo_; }
vec2 med9v(vec2 p0, vec2 p1, vec2 p2, vec2 p3, vec2 p4, vec2 p5, vec2 p6, vec2 p7, vec2 p8) {
    V2_SORT(p1, p2); V2_SORT(p4, p5); V2_SORT(p7, p8);
    V2_SORT(p0, p1); V2_SORT(p3, p4); V2_SORT(p6, p7);
    V2_SORT(p1, p2); V2_SORT(p4, p5); V2_SORT(p7, p8);
    V2_SORT(p0, p3); V2_SORT(p5, p8); V2_SORT(p4, p7);
    V2_SORT(p3, p6); V2_SORT(p1, p4); V2_SORT(p2, p5);
    V2_SORT(p4, p7); V2_SORT(p4, p2); V2_SORT(p6, p4);
    V2_SORT(p4, p2);
    return p4;
}
vec2 rawF(int off, ivec2 b, int gw, int gh) {
    b = clamp(b, ivec2(0), ivec2(gw - 1, gh - 1));
    return field[off + b.y * gw + b.x];
}
vec2 median3At(int off, ivec2 b, int gw, int gh) {
    return med9v(rawF(off, b + ivec2(-1, -1), gw, gh), rawF(off, b + ivec2(0, -1), gw, gh), rawF(off, b + ivec2(1, -1), gw, gh),
                 rawF(off, b + ivec2(-1, 0), gw, gh), rawF(off, b, gw, gh), rawF(off, b + ivec2(1, 0), gw, gh),
                 rawF(off, b + ivec2(-1, 1), gw, gh), rawF(off, b + ivec2(0, 1), gw, gh), rawF(off, b + ivec2(1, 1), gw, gh));
}
