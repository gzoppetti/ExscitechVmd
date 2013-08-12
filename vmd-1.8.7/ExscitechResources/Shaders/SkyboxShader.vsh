// GLSL version 1.50
#version 150

// Attributes
in vec3 g_position;
out vec3 g_texCoord0;

// Uniforms
uniform mat4 g_view;
uniform mat4 g_projection;
uniform mat4 g_world;

void
main (void)
{
  mat4 worldViewProjection = g_projection * g_view * g_world;
  gl_Position = worldViewProjection * vec4 (g_position, 1);
  g_texCoord0 = g_position * -1;
}