#version 460 core
layout (location = 0) out vec4 outColor;
layout (location = 2) in vec2 TexCoords;
#ifdef VULKAN
layout (binding = 2) uniform sampler2D text;
layout (push_constant) uniform pushColor { vec4 textColor; } pc;
#define TEXT_COLOR pc.textColor
#else
layout (binding = 0) uniform sampler2D text;
layout (location = 4) uniform vec4 textColor;
#define TEXT_COLOR textColor
#endif

void main()
{
  vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
  outColor = TEXT_COLOR * sampled;
}
