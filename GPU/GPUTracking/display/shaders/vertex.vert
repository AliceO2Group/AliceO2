/*#version 450

layout(location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
*/

#version 450 core
layout (location = 0) in vec3 pos;
layout (binding = 0) uniform UniformBufferObject { mat4 ModelViewProj; } ubo;

void main()
{
  gl_Position = ubo.ModelViewProj * vec4(pos.x, pos.y, pos.z, 1.0);
}
