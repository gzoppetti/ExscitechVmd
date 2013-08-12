#version 150

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out vec4 eye_position;
out vec2 g_texCoord;
out vec4 g_fragmentDetail;

uniform vec4 g_detail;
uniform mat4 g_projection;

void main()
{
	vec4 pos = gl_in[0].gl_Position;
	float r = g_detail.w;
	
    eye_position = pos;
	
    // Vertex 1
    g_texCoord = vec2(-1.0,-1.0);
    gl_Position = pos;
    gl_Position.x -= r;
    gl_Position.y -= r;
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 2
    g_texCoord = vec2(1.0,-1.0);
    gl_Position = pos;
    gl_Position.x += r;
    gl_Position.y -= r;
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 3
    g_texCoord = vec2(-1.0, 1.0);
    gl_Position = pos;
    gl_Position.x -= r;
    gl_Position.y += r;
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    // Vertex 4
    g_texCoord = vec2(1.0,1.0);
    gl_Position = pos;
    gl_Position.x += r;
    gl_Position.y += r;
    gl_Position = g_projection  * gl_Position;
    EmitVertex();

    EndPrimitive();
}