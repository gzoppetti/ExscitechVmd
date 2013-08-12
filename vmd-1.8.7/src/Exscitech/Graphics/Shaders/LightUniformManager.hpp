#ifndef LIGHTUNIFORMMANAGER_HPP_
#define LIGHTUNIFORMMANAGER_HPP_

#include "Exscitech/Display/Camera.hpp"
#include "Exscitech/Graphics/Lighting/Material.hpp"

namespace Exscitech
{
  class LightUniformManager
  {
  public:

    LightUniformManager(int shaderId, int numberOfLights);

    void
    setUniforms(Camera* camera, Material* material);

  private:

    void
    obtainUniformLocations(int shaderId);

    int
    getUniformLocation(int shaderId, const std::string& attribute);

  private:

    enum BaseIndexEnum
    {
      AMBIENT_REFLECTION,
      DIFFUSE_REFLECTION,
      SPECULAR_REFLECTION,
      SPECULAR_POWER,
      EMISSIVE_INTENSITY,
      NUM_LIGHTS,
      EYE_POSITION,
      AMBIENT_INTENSITY,
      NUM_BASE_UNIFORMS
    };

    enum LightIndexEnum
    {
      LIGHT_TYPE,
      LIGHT_POSITION,
      LIGHT_DIFFUSE_INTENSITY,
      LIGHT_SPECULAR_INTENSITY,
      LIGHT_ATTENUATION_COEFFICIENTS,
      LIGHT_DIRECTION,
      LIGHT_CUTOFF_COS_ANGLE,
      LIGHT_FALLOFF,
      NUM_LIGHT_UNIFORMS
    };

  private:

  private:
    typedef int LightParameterBlock[NUM_LIGHT_UNIFORMS];

  private:
    int m_baseParameters[NUM_BASE_UNIFORMS];
    LightParameterBlock* m_lightParameters;
    int m_numberOfLights;
  };
}


#endif
