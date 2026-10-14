#version 460 core
layout (location = 0) in vec3 pos;
#ifdef VULKAN
layout (binding = 0) uniform uniformMatrix { mat4 ModelViewProj; } um;
layout (push_constant) uniform pushSize { layout(offset = 16) float size; } ps;
#define MODEL_VIEW_PROJ um.ModelViewProj
#define POINT_SIZE ps.size
#else
layout (location = 0) uniform mat4 ModelViewProj;
layout (location = 5) uniform float size;
#define MODEL_VIEW_PROJ ModelViewProj
#define POINT_SIZE size
#endif

void main()
{
  gl_Position = MODEL_VIEW_PROJ * vec4(pos.x, pos.y, pos.z, 1.0);
  gl_PointSize = POINT_SIZE;
}
