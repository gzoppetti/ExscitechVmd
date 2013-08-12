#include <sstream>

#include "Exscitech/Graphics/Shaders/LightUniformManager.hpp"
#include "Exscitech/Graphics/Lighting/Material.hpp"
#include "Exscitech/Graphics/Lighting/Light.hpp"
#include "Exscitech/Graphics/Lighting/PointLight.hpp"
#include "Exscitech/Graphics/Lighting/DirectionalLight.hpp"
#include "Exscitech/Graphics/Lighting/SpotLight.hpp"

namespace Exscitech
{
  LightUniformManager::LightUniformManager (int shaderId, int numberOfLights) :
      m_numberOfLights (numberOfLights)
  {
    m_lightParameters = new LightParameterBlock[m_numberOfLights];

    obtainUniformLocations (shaderId);
  }

  void
  LightUniformManager::setUniforms (Camera* camera, Material* material)
  {
    glUniform4fv (m_baseParameters[AMBIENT_REFLECTION], 1,
        &material->getAmbientColor ()[0]);

    glUniform4fv (m_baseParameters[DIFFUSE_REFLECTION], 1,
        &material->getDiffuseColor ()[0]);

    glUniform4fv (m_baseParameters[SPECULAR_REFLECTION], 1,
        &material->getSpecularColor ()[0]);
    glUniform1f (m_baseParameters[SPECULAR_POWER], material->getShininess ());
    glUniform4fv (m_baseParameters[EMISSIVE_INTENSITY], 1,
        &material->getEmissionColor ()[0]);

    glUniform1i (m_baseParameters[NUM_LIGHTS], m_numberOfLights);
    glUniform3fv (m_baseParameters[EYE_POSITION], 1,
        &camera->getPosition ()[0]);

    // Default GL light model base ambient
    //Vector4f totalAmbient (0.2f, 0.2f, 0.2f, 1);
    Vector4f totalAmbient;

    int index = 0;
    std::vector<Light*> lights = Light::getLights ();

    for (int i = 0; i < m_numberOfLights; ++i)
    {
      if (static_cast<uint> (i) < lights.size ())
      {
        Light* light = lights[i];
        std::ostringstream stringBuilder;
        stringBuilder << "g_lights[" << index << "].";
        std::string lightsIndex = stringBuilder.str ();
        glUniform1i (m_lightParameters[i][LIGHT_TYPE], light->getLightType ());

        switch (light->getLightType ())
        {
          case Light::POINT_LIGHT:
          {
            PointLight* pointLight = static_cast<PointLight*> (light);
            glUniform3fv (m_lightParameters[i][LIGHT_POSITION], 1,
                &pointLight->getPosition ()[0]);

            glUniform3fv (m_lightParameters[i][LIGHT_ATTENUATION_COEFFICIENTS],
                1, &pointLight->getAttenuation ()[0]);
            break;
          }
          case Light::SPOT_LIGHT:
          {
            SpotLight* spotLight = static_cast<SpotLight*> (light);
            glUniform3fv (m_lightParameters[i][LIGHT_POSITION], 1,
                &spotLight->getPosition ()[0]);
            glUniform3fv (m_lightParameters[i][LIGHT_ATTENUATION_COEFFICIENTS],
                1, &spotLight->getAttenuation ()[0]);
            glUniform3fv (m_lightParameters[i][LIGHT_DIRECTION], 1,
                &spotLight->getSpotDirection ()[0]);
            glUniform1f (m_lightParameters[i][LIGHT_CUTOFF_COS_ANGLE],
                spotLight->getSpotCutoff ());
            glUniform1f (m_lightParameters[i][LIGHT_FALLOFF],
                spotLight->getSpotExponent ());
            break;
          }
          case Light::DIRECTIONAL_LIGHT:
          {
            DirectionalLight* directionalLight =
                static_cast<DirectionalLight*> (light);
            glUniform3fv (m_lightParameters[i][LIGHT_DIRECTION], 1,
                &directionalLight->getDirection ()[0]);
            break;
          }
          default:
            break;
        }

        glUniform4fv (m_lightParameters[i][LIGHT_DIFFUSE_INTENSITY], 1,
            &light->getDiffuseColor ()[0]);
        glUniform4fv (m_lightParameters[i][LIGHT_SPECULAR_INTENSITY], 1,
            &light->getSpecularColor ()[0]);
        totalAmbient += light->getAmbientColor ();
      }
    }

    glUniform4fv (m_baseParameters[AMBIENT_INTENSITY], 1, &totalAmbient[0]);
  }

  void
  LightUniformManager::obtainUniformLocations (int shaderId)
  {

    m_baseParameters[AMBIENT_REFLECTION] = glGetUniformLocation (shaderId,
        "g_ambientReflection");
    m_baseParameters[DIFFUSE_REFLECTION] = glGetUniformLocation (shaderId,
        "g_diffuseReflection");
    m_baseParameters[SPECULAR_REFLECTION] = glGetUniformLocation (shaderId,
        "g_specularReflection");
    m_baseParameters[SPECULAR_POWER] = glGetUniformLocation (shaderId,
        "g_specularPower");
    m_baseParameters[EMISSIVE_INTENSITY] = glGetUniformLocation (shaderId,
        "g_emissiveIntensity");
    m_baseParameters[NUM_LIGHTS] = glGetUniformLocation (shaderId,
        "g_numLights");
    m_baseParameters[EYE_POSITION] = glGetUniformLocation (shaderId,
        "g_eyePosition");
    m_baseParameters[AMBIENT_INTENSITY] = glGetUniformLocation (shaderId,
        "g_ambientIntensity");

    for (int i = 0; i < m_numberOfLights; ++i)
    {
      std::ostringstream stringBuilder;
      stringBuilder << "g_lights[" << i << "].";
      std::string lightsIndex = stringBuilder.str ();
      m_lightParameters[i][LIGHT_TYPE] = getUniformLocation (shaderId,
          lightsIndex + "type");
      m_lightParameters[i][LIGHT_POSITION] = getUniformLocation (shaderId,
          lightsIndex + "position");
      m_lightParameters[i][LIGHT_DIFFUSE_INTENSITY] = getUniformLocation (
          shaderId, lightsIndex + "diffuseIntensity");
      m_lightParameters[i][LIGHT_SPECULAR_INTENSITY] = getUniformLocation (
          shaderId, lightsIndex + "specularIntensity");
      m_lightParameters[i][LIGHT_ATTENUATION_COEFFICIENTS] =
          getUniformLocation (shaderId,
              lightsIndex + "attenuationCoefficients");
      m_lightParameters[i][LIGHT_DIRECTION] = getUniformLocation (shaderId,
          lightsIndex + "direction");
      m_lightParameters[i][LIGHT_CUTOFF_COS_ANGLE] = getUniformLocation (
          shaderId, lightsIndex + "cutoffCosAngle");
      m_lightParameters[i][LIGHT_FALLOFF] = getUniformLocation (shaderId,
          lightsIndex + "falloff");
    }
  }

  int
  LightUniformManager::getUniformLocation( int shaderId, const std::string& attribute)
  {
    return glGetUniformLocation(shaderId, attribute.c_str());
  }
}
