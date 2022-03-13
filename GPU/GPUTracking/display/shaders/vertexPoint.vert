#version 460 core
layout (location = 0) in vec3 pos;
layout (binding = 0) uniform uniformMatrix { mat4 ModelViewProj; } um;
layout (push_constant) uniform pushSize { vec4 color; float size; } ps;

void main()
{
  gl_Position = um.ModelViewProj * vec4(pos.x, pos.y, pos.z, 1.0);
  gl_PointSize = ps.size;
}
