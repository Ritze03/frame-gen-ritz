// Histogram increment; the host blends ONE,ONE into the 256x2 R32F hist (LAYOUT 3.5).
layout(location = 0) out vec4 oCount;
void main() {
    oCount = vec4(1.0, 0.0, 0.0, 0.0);
}
