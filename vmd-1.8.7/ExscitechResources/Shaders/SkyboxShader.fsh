#version 150

in vec3 g_texCoord0;
out vec4 g_fragColor;

uniform samplerCube g_sampler0;

void 
main (void)
{
  g_fragColor = texture(g_sampler0, g_texCoord0);
}
