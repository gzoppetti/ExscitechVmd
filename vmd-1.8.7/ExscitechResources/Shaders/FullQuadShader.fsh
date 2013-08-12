#version 400

in vec2 texCoord;

out vec4 g_fragColor;

uniform sampler2D g_sampler;

#define DEPTH_DEBUG 0

#if DEPTH_DEBUG
float readDepth( in vec2 coord ) 
{
	vec2 cameraRange = vec2(0.01, 100.0);
	return (2.0 * cameraRange.x) / (cameraRange.y + cameraRange.x - texture2D( g_sampler, coord ).x * (cameraRange.y - cameraRange.x));	
}

void
main(void)
{
	float depth = readDepth(texCoord);
    g_fragColor = vec4(depth);
}

#else
void
main(void)
{
    g_fragColor = texture2D(g_sampler, texCoord);
}

#endif