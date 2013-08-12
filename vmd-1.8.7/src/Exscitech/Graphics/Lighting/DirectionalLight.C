#include <GL/glew.h>

#include "Exscitech/Graphics/Lighting/DirectionalLight.hpp"

namespace Exscitech
{
  const Vector3f DirectionalLight::DEFAULT_DIRECTION (0, 0, 1);

  DirectionalLight::DirectionalLight (uchar lightNumber) :
        Light (lightNumber),
        m_direction (DEFAULT_DIRECTION)
  {
  }

  DirectionalLight::~DirectionalLight ()
  {
  }

  void
  DirectionalLight::setDirection (const Vector3f& direction)
  {
    m_direction = direction;
  }

  Vector3f
  DirectionalLight::getDirection () const
  {
    return (m_direction);
  }

  Light::LightType
  DirectionalLight::getLightType () const
  {
    return (Light::DIRECTIONAL_LIGHT);
  }
}

