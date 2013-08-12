
// GLSL Version 1.50
#version 150

// Input from vertex shader
in  vec4 g_vertexColor;

out vec4 g_fragColor;

void 
main (void)
{
  g_fragColor = g_vertexColor;    
}
