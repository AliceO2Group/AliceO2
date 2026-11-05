#version 460 core
layout (location = 0) in vec4 vertex;
layout (location = 2) out vec2 TexCoords;
#ifdef VULKAN
layout (binding = 0) uniform uniformMatrix { mat4 projection; } um;
#define PROJECTION um.projection
#else
layout (location = 0) uniform mat4 projection;
#define PROJECTION projection
#endif

void main()
{
  gl_Position = PROJECTION * vec4(vertex.xy, 0.0, 1.0);
  TexCoords = vertex.zw;
}
