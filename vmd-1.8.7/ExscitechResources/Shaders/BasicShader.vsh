
#version 150
// Place Drawable's transform in this variable
in vec3 g_position;
out vec4 g_color;

uniform mat4 g_view;
uniform mat4 g_projection;
uniform mat4 g_world;
uniform vec4 g_inColor;

void
main (void)
{               
    mat4 worldViewProj = g_projection * g_view * g_world;
    
    gl_Position = worldViewProj * vec4 (g_position, 1);
    
    g_color = g_inColor;
}

