#version 460 core
layout (location = 0) out vec4 outColor;
layout (push_constant) uniform pushColor { vec4 color; } pc;

void main()
{
    outColor = vec4(pc.color.x, pc.color.y, pc.color.z, 1.f);
}
