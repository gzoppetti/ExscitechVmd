#version 150

layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

in float halfRadius[];
in vec4 pos1[];
in vec4 pos2[];
in vec3 color[];

out vec4 eye_position;
out vec2 g_texCoord;
out vec3 g_fragmentColor;
out float zDelta;
out float g_radius;

uniform mat4 g_projection;

void main()
{
    eye_position = pos1[0];
	g_fragmentColor = color[0] + vec3(0.2, 0.2, 0.2);	
	
	zDelta = pos2[0].z - pos1[0].z;
    g_radius = halfRadius[0];
    
    // Vertex 1
    g_texCoord = vec2(0,1.0);
    gl_Position = pos1[0];
    gl_Position.y += halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 2
    g_texCoord = vec2(0,-1.0);
    gl_Position = pos1[0];
    gl_Position.y -= halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 3
    g_texCoord = vec2(1.0,1.0);
    gl_Position = pos2[0];
    gl_Position.y += halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 4
    g_texCoord = vec2(1.0,-1.0);
    gl_Position = pos2[0];
    gl_Position.y -= halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    EndPrimitive();
    
    // Vertex 1
    g_texCoord = vec2(0,1.0);
    gl_Position = pos1[0];
    gl_Position.x += halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 2
    g_texCoord = vec2(0,-1.0);
    gl_Position = pos1[0];
    gl_Position.x -= halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 3
    g_texCoord = vec2(1.0,1.0);
    gl_Position = pos2[0];
    gl_Position.x += halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 4
    g_texCoord = vec2(1.0,-1.0);
    gl_Position = pos2[0];
    gl_Position.x -= halfRadius[0];
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    EndPrimitive();
}