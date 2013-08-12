// GLSL version 1.50
#version 150

uniform int g_numLights;

struct Light
{
  // 0 if directional, 1 if point, 2 if spot
  int type;

  // All lights have these parameters
  vec4 diffuseIntensity;
  vec4 specularIntensity;

  // Point and spot light parameters
  vec3 position;
  vec3 attenuationCoefficients;

  // Directional and spot light parameter
  vec3 direction;

  // Spot light parameters
  float cutoffCosAngle;
  float falloff;
};

const int MAX_LIGHTS = 4;
uniform Light g_lights[MAX_LIGHTS];

// Material properties
uniform vec4 g_ambientIntensity;
uniform vec4 g_ambientReflection;
uniform vec4 g_diffuseReflection;
uniform vec4 g_specularReflection;
uniform float g_specularPower;
uniform vec4 g_emissiveIntensity;

// Attributes
in vec3 g_position;
in vec3 g_normal;
in vec2 g_tex0;

out vec4 g_vertexColor;
out vec2 g_fragTex0;

// Uniforms
uniform mat4 g_view;
uniform mat4 g_projection;
uniform mat4 g_world;
uniform vec3 g_eyePosition;
// **

// Calculate diffuse and specular lighting for a single light
vec4
calculateLighting (Light light, vec3 vertexPosition, vec3 vertexNormal);

// **

void
main (void)
{ 
  g_fragTex0 = g_tex0;
  
  vec4 worldPosition = g_world * vec4(g_position, 1);
  vec4 distanceVector = worldPosition - vec4(g_eyePosition, 1);
  float distanceAwayFromCamera = length(distanceVector);
  
  mat4 worldViewProjection = g_projection * g_view * g_world;
  
  // Transform vertex into clip space
  gl_Position = worldViewProjection * vec4 (g_position, 1);
  
  // Transform vertex into world space for lighting
  vec3 positionWorld = vec3 (g_world * vec4 (g_position, 1));

  mat3 normalTransform = mat3 (g_world);
  normalTransform = inverse (normalTransform);
  normalTransform = transpose (normalTransform);
  // Normal transform is world inverse transpose
  vec3 normalWorld = normalTransform * g_normal;
  // Bring normal into world space
  normalWorld = normalize (normalWorld);

  // Handle ambient and emissive light
  //   It's independent of any particular light
  g_vertexColor = g_ambientReflection * g_ambientIntensity
      + g_emissiveIntensity;
  // Iterate over all lights and calculate diffuse and specular contributions
  for (int i = 0; i < g_numLights; ++i)
  {
    g_vertexColor
        += calculateLighting (g_lights[i], positionWorld, normalWorld);
  }
  // Stay in bounds [0, 1]
  g_vertexColor = clamp (g_vertexColor, 0.0, 1.0);
}

// **

vec4
calculateLighting (Light light, vec3 vertexPosition, vec3 vertexNormal)
{
  // Light vector points toward the light
  vec3 lightVector;
  if (light.type == 0)
  { // Directional
    lightVector = normalize (-light.direction);
  }
  else
  { // Point or spot
    lightVector = normalize (light.position - vertexPosition);
  }
  // Light intensity is proportional to angle between light vector
  //   and vertex normal
  float lambertianCoef = max (dot (lightVector, vertexNormal), 0.0f);
  vec4 diffuseAndSpecular = vec4 (0.0f);
  if (lambertianCoef != 0.0)
  {
    // Light is incident on vertex, not shining on its edge or back
    vec4 diffuseColor = g_diffuseReflection * light.diffuseIntensity;
    diffuseColor *= lambertianCoef;

    vec4 specularColor = g_specularReflection * light.specularIntensity;
    // See how light reflects off of vertex
    vec3 reflectionVector = reflect (lightVector, vertexNormal);
    // Where is the eye?
    vec3 eyeVector = normalize (g_eyePosition - vertexPosition);
    // Light intensity is proportional to angle between reflection vector
    //   and eye vector
    float specularCoef = max (dot (eyeVector, reflectionVector), 0.0f);
    // Material's specular power determines size of bright spots
    specularColor *= pow (specularCoef, g_specularPower);

    float attenuation = 1.0f;
    if (light.type != 0)
    { // Non-directional, so light attenuates
      float distance = length (vertexPosition - light.position);
      attenuation = 1.0f / (light.attenuationCoefficients.x
          + light.attenuationCoefficients.y * distance
          + light.attenuationCoefficients.z * distance * distance);
    }
    float spotFactor = 1.0f;
    if (light.type == 2)
    { // Spot light
      float cosTheta = dot (-lightVector, light.direction);
      cosTheta = max (cosTheta, 0.0f);
      spotFactor = (cosTheta >= light.cutoffCosAngle) ? cosTheta : 0.0f;
      spotFactor = pow (spotFactor, light.falloff);
    }
    diffuseAndSpecular = spotFactor * attenuation * (diffuseColor
        + specularColor);
  }

  return (diffuseAndSpecular);
}

