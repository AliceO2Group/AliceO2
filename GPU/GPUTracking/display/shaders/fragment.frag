#version 460 core
layout (location = 0) out vec4 outColor;
#ifdef VULKAN
layout (push_constant) uniform pushColor { vec4 color; } pc;
#define COLOR pc.color
#else
layout (location = 4) uniform vec4 color;
#define COLOR color
#endif

void main()
{
  outColor = vec4(COLOR.x, COLOR.y, COLOR.z, 1.f);
}
