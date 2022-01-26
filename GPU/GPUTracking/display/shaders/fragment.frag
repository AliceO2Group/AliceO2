/*#version 450

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
*/

#version 450 core
layout(location = 0) out vec4 outColor;
//uniform vec3 color;

void main()
{
    //outColor = vec4(color.x, color.y, color.z, 1.f);
    outColor = vec4(1.f, 1.f, 1.f, 1.f);
}
