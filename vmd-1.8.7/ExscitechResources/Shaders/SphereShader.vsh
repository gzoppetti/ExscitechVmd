#version 150

in vec3 g_point;

uniform mat4 g_view;
uniform mat4 g_world;

void
main (void)
{                  
    mat4 worldView = g_view * g_world;
    gl_Position = worldView * vec4 (g_point, 1);
}