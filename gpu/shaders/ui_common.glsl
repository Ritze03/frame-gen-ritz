// Shared declarations of the static-UI-protection passes (ui_*.comp). Own descriptor set layout and push
// constants (not common.glsl's): the passes run in recordObserve/recordSynth, beside the interpolation passes.
//
// binding 0: sampled frame (observe: the new real frame; patch: the real `curr`). Only texelFetch.
// binding 1: storage image, declared per shader (observe: the clean copy; patch: the generated frame).
// binding 2: S = per-pixel state words (box: pixel; whole screen: cell). Word = r | g << 8 | b << 16 | count << 24.
// binding 3: M = per-frame mask buffer. Header words [0..3]: [0] = number of still cells (whole screen);
//            then, whole screen only, one flag word per 32x32 tile; then the mask at maskOff: 4 pixels per word
//            (byte k = pixel 4x + k, 0 = none, 255 = full, 128 = feather), maskStride words per row, covering the
//            rectangle (maskX, maskY, maskW, maskH) of the frame.

layout(set = 0, binding = 0) uniform sampler2D src;
layout(set = 0, binding = 2, std430) buffer StateBuf { uint st[]; };
layout(set = 0, binding = 3, std430) buffer MaskBuf { uint mk[]; };

// == framegen.cpp's UiPC, field for field
layout(push_constant) uniform PC {
    int W, H;                 // frame
    int boxX, boxY, boxW, boxH;   // the full-resolution crosshair box
    int cell, cw, ch;         // whole screen: detection cell (pixels per side) and the cell grid
    int wholeScreen;
    int still;                // frames a pixel/cell must stay still
    float thr;                // stability threshold, 8-bit levels
    int reach;                // inpaint reach
    int reset;                // the state words describe no previous frame: start every counter at 0
    int tilesX, tilesY;       // 32x32-pixel tiles
    int stillLimit;           // still cells above this = whole-screen passes suppressed for this frame
    int maskX, maskY, maskW, maskH, maskStride, maskOff;
    int gate;                 // 1: tile flags + suppression apply (whole screen)
} pc;

const int TILE = 32;

uint maskByte(ivec2 rel) {   // rel = pixel relative to the mask rectangle
    return (mk[pc.maskOff + rel.y * pc.maskStride + (rel.x >> 2)] >> (8 * (rel.x & 3))) & 255u;
}

// whole-screen gate: tile flagged and the frame not suppressed. tile = pixel / TILE.
bool tileLive(ivec2 tile) {
    return mk[0] <= uint(pc.stillLimit) && mk[4 + tile.y * pc.tilesX + tile.x] != 0u;
}
