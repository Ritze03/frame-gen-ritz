// Global median seed candidate from the 256x2 histogram (x row 0, y row 1). Writes 2 * median.
// medhist.comp:10-37; PY:142.
uniform sampler2D uHist;
uniform int uCount;   // bw * bh of the fine level
layout(location = 0) out vec4 oMedian;

float kth(int c, uint k) {
    float acc = 0.0;
    for (int b = 0; b < 256; b++) {
        acc += texelFetch(uHist, ivec2(b, c), 0).r;
        if (acc > float(k)) return float(b - 128);
    }
    return 127.0;
}

void main() {
    uint k1 = uint(uCount - 1) / 2u, k2 = uint(uCount) / 2u;
    float mx = 0.5 * (kth(0, k1) + kth(0, k2));
    float my = 0.5 * (kth(1, k1) + kth(1, k2));
    oMedian = vec4(2.0 * mx, 2.0 * my, 0.0, 0.0);
}
