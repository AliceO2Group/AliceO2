#version 460 core
layout (location = 0) in vec4 vertex;
layout (location = 2) out vec2 TexCoords;
layout (binding = 0) uniform uniformMatrix { mat4 projection; } um;

void main()
{
  gl_Position = um.projection * vec4(vertex.xy, 0.0, 1.0);
  TexCoords = vertex.zw;
}
