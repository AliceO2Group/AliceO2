#version 460 core
layout (location = 0) out vec4 outColor;
layout (binding = 1) uniform uniformColor { vec4 color; } uc;

void main()
{
    outColor = vec4(uc.color.x, uc.color.y, uc.color.z, 1.f);
}
