#version 450 core
layout (location = 0) out vec4 outColor;
//layout (binding = 1) uniform UniformBufferObject { vec3 color; } ubo;
layout (push_constant) uniform UniformBufferObject { vec3 color; } ubo;

void main()
{
    outColor = vec4(ubo.color.x, ubo.color.y, ubo.color.z, 1.f);
}
