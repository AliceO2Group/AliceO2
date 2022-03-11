#version 460 core
layout (location = 0) in vec3 pos;
layout (binding = 0) uniform UniformBufferObject { mat4 ModelViewProj; } ubo;

void main()
{
  gl_Position = ubo.ModelViewProj * vec4(pos.x, pos.y, pos.z, 1.0);
  gl_PointSize = 1.0;
}
