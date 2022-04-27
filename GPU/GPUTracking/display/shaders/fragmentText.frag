#version 460 core
layout (location = 0) out vec4 outColor;
layout (location = 2) in vec2 TexCoords;
layout (binding = 2) uniform sampler2D text;
layout (push_constant) uniform pushColor { vec4 textColor; } pc;

void main()
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    outColor = pc.textColor * sampled;
}
