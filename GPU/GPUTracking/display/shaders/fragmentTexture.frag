#version 460 core
layout (location = 0) out vec4 outColor;
layout (location = 2) in vec2 TexCoords;
layout (binding = 2) uniform sampler2D tex;
layout (push_constant) uniform pushAlpha { float alpha; } pa;

void main()
{
    outColor = vec4(texture(tex, TexCoords).rgb, pa.alpha);
}
