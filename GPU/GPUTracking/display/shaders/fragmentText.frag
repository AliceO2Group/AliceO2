#version 460 core
layout (location = 0) out vec4 outColor;
layout (location = 2) in vec2 TexCoords;
layout (binding = 2) uniform sampler2D text;
layout (push_constant) uniform UniformBufferObject { vec4 textColor; } ubo;

void main()
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    outColor = ubo.textColor * sampled;
}
