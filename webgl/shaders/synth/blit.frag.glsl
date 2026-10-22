// synth/blit: presentation only (not in the reference). Copies an RGBA8 frame into the target viewport.
// uFlipY compensates the bottom-up default framebuffer (canvas). uExact (viewport size == frame size) is a
// bit-exact texelFetch copy; otherwise bilinear via the LINEAR filter of uSrc.
uniform sampler2D uSrc;     // RGBA8, LINEAR
uniform vec2 uVpOrigin;     // target viewport origin in pixels
uniform vec2 uVpSize;       // target viewport size in pixels
uniform int uFlipY;
uniform int uExact;

layout(location = 0) out vec4 oColor;

void main() {
    vec2 c = gl_FragCoord.xy - uVpOrigin;   // pixel index in the viewport + 0.5
    if (uFlipY != 0) c.y = uVpSize.y - c.y;
    if (uExact != 0) {
        ivec2 sz = textureSize(uSrc, 0);
        oColor = vec4(texelFetch(uSrc, clamp(ivec2(c), ivec2(0), sz - 1), 0).rgb, 1.0);
    } else {
        oColor = vec4(texture(uSrc, c / uVpSize).rgb, 1.0);
    }
}
