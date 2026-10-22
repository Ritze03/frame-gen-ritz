// Global median seed candidate (medhist.comp:10-37; PY:142) from the cumulative banded histogram of estimate/medhist (64 x (2 B),
// 4 bins per texel, component c's bands in rows [c B, c B + B)). The k-th element is the first bin whose cumulative count exceeds k (a binary search
// on the monotone counts); median = mean of the k1-th and k2-th, written x 2.
uniform sampler2D uHist;
uniform int uCount;   // bw * bh of the fine level
layout(location = 0) out vec4 oMedian;

float cum(int c, int B, int b) {
    float s = 0.0;
    for (int j = 0; j < B; j++) s += texelFetch(uHist, ivec2(b >> 2, c * B + j), 0)[b & 3];
    return s;
}
float kth(int c, float k) {
    int B = textureSize(uHist, 0).y >> 1;
    if (!(cum(c, B, 255) > k)) return 127.0;
    int lo = 0, hi = 255;   // invariant: cum(hi) > k
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cum(c, B, mid) > k) hi = mid; else lo = mid + 1;
    }
    return float(lo - 128);
}

void main() {
    float k1 = float(uint(uCount - 1) / 2u), k2 = float(uint(uCount) / 2u);
    float mx = 0.5 * (kth(0, k1) + kth(0, k2));
    float my = 0.5 * (kth(1, k1) + kth(1, k2));
    oMedian = vec4(2.0 * mx, 2.0 * my, 0.0, 0.0);
}
