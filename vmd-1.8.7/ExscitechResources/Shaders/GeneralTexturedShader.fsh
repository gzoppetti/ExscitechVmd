
// GLSL Version 1.50
#version 150

// Input from vertex shader
in  vec4 g_vertexColor;
in vec2 g_fragTex0;
in float g_fogPercentage;

out vec4 g_fragColor;

//Uniforms
uniform sampler2D sampler0;

void 
main (void)
{
  	g_fragColor = texture(sampler0, g_fragTex0) + g_vertexColor;
}
