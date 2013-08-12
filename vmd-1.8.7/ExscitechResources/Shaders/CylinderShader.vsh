#version 150

in vec3 g_point1;
in vec3 g_point2;

// XYZ color W radius

out float halfRadius;
out vec4 pos1;
out vec4 pos2;
out vec3 color;

uniform mat4 g_view;
uniform mat4 g_projection;
uniform mat4 g_world;
uniform vec4 g_details;

void
main (void)
{                  
    mat4 worldView = g_view * g_world;
    pos1 = worldView * vec4(g_point1, 1);
    pos2 = worldView * vec4(g_point2, 1);
    halfRadius = g_details.w * length(g_world[0].xyz);
    color = g_details.xyz;
}	

