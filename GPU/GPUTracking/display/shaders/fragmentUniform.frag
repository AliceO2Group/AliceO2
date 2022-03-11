#version 460 core
layout (location = 0) out vec4 outColor;
layout (binding = 1) uniform UniformBufferObject { vec4 color; } ubo;

void main()
{
    outColor = vec4(ubo.color.x, ubo.color.y, ubo.color.z, 1.f);
}
