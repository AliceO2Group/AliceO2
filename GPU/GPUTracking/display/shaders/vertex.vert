#version 460 core
layout (location = 0) in vec3 pos;
#ifdef VULKAN
layout (binding = 0) uniform uniformMatrix { mat4 ModelViewProj; } um;
#define MODEL_VIEW_PROJ um.ModelViewProj
#else
layout (location = 0) uniform mat4 ModelViewProj;
#define MODEL_VIEW_PROJ ModelViewProj
#endif

void main()
{
  gl_Position = MODEL_VIEW_PROJ * vec4(pos.x, pos.y, pos.z, 1.0);
}
