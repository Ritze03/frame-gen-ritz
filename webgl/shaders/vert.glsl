#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
// Fullscreen triangle from gl_VertexID (drawArrays(TRIANGLES, 0, 3)); no attributes, no varyings.
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
