#version 460 core
layout (location = 0) out vec4 outColor;
layout (location = 2) in vec2 TexCoords;
#ifdef VULKAN
layout (binding = 2) uniform sampler2D tex;
layout (push_constant) uniform pushAlpha { float alpha; } pa;
#define ALPHA pa.alpha
#else
layout (binding = 0) uniform sampler2D tex;
layout (location = 4) uniform float alpha;
#define ALPHA alpha
#endif

void main()
{
  outColor = vec4(texture(tex, TexCoords).rgb, ALPHA);
}
