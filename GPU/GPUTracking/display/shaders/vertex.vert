#version 460 core
layout (location = 0) in vec3 pos;
layout (binding = 0) uniform uniformMatrix { mat4 ModelViewProj; } um;

void main()
{
  gl_Position = um.ModelViewProj * vec4(pos.x, pos.y, pos.z, 1.0);
}
