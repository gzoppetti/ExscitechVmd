#version 150

in vec3 g_position;

out vec2 texCoord;

void
main (void)
{
    gl_Position = vec4(g_position, 1);
    texCoord = (g_position.xy + vec2(1, 1)) / 2.0;
}