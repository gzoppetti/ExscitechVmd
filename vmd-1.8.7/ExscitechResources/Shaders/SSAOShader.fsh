#version 150

uniform sampler2D depthTexture;
uniform sampler2D randomTexture;

uniform vec2 cameraRange;
uniform vec2 screenSize;

in vec2 texCoord;

out vec4 g_fragColor;
  	
float readDepth( in vec2 coord ) 
{
	return (2.0 * cameraRange.x) / (cameraRange.y + cameraRange.x - texture2D( depthTexture, coord ).x * (cameraRange.y - cameraRange.x));	
}

vec3 normalFromDepth(in float depth, in vec2 texCoords)
{
	const vec2 offset1 = vec2(0.0, 0.005);
	const vec2 offset2 = vec2(0.005, 0.0);
	
	float depth1 = readDepth(offset1);
	float depth2 = readDepth(offset2);
	
	vec3 p1 = vec3(offset1, depth1 - depth);
	vec3 p2 = vec3(offset2, depth2 - depth);
	
	vec3 normal = cross(p1, p2);
	normal.z = -normal.z;
	
	return normalize(normal);
}

const float totalStrength = 1.0;
const float base = 0.0;
	
const float area = 0.075;
const float falloff = 0.001;
	
const float radius = 0.01;
	
const int samples = 16;
uniform vec3 sampleSphere[samples] = vec3[samples](
      vec3( 0.5381, 0.1856,-0.4319), vec3( 0.1379, 0.2486, 0.4430),
      vec3( 0.3371, 0.5679,-0.0057), vec3(-0.6999,-0.0451,-0.0019),
      vec3( 0.0689,-0.1598,-0.8547), vec3( 0.0560, 0.0069,-0.1843),
      vec3(-0.0146, 0.1402, 0.0762), vec3( 0.0100,-0.1924,-0.0344),
      vec3(-0.3577,-0.5301,-0.4358), vec3(-0.3169, 0.1063, 0.0158),
      vec3( 0.0103,-0.5869, 0.0046), vec3(-0.0897,-0.4940, 0.3287),
      vec3( 0.7119,-0.0154,-0.0918), vec3(-0.0533, 0.0596,-0.5411),
      vec3( 0.0352,-0.0631, 0.5460), vec3(-0.4776, 0.2847,-0.0271));
      
void main(void)
{      
    float depth = readDepth(texCoord);
    if (depth > 0.99)
    	discard;
	vec3 random = normalize( texture2D(randomTexture, texCoord).rgb);
	
	vec3 position = vec3(texCoord, depth);
	vec3 normal = normalFromDepth(depth, texCoord);
	float radiusDepth = radius / depth;
	float occlusion = 0.0;
	for(int i = 0; i < samples; i++)
	{
		vec3 ray = radiusDepth * reflect(sampleSphere[i], random);
		vec3 hemiRay = position + sign(dot(ray, normal)) * ray;
		
		float occDepth = readDepth(clamp(hemiRay.xy, 0.0, 1.0));
		float difference = depth - occDepth;
		
		occlusion += step(falloff, difference) * (1.0 - smoothstep(falloff, area, difference));
	}
	
	float ao = 1.0 - totalStrength * occlusion * (1.0 / samples);
	g_fragColor = vec4(clamp(ao + base, 0.0, 1.0));
}