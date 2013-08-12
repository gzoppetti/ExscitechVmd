
#include "Exscitech/Graphics/Lighting/PointLight.hpp"

namespace Exscitech
{
  const Vector3f PointLight::DEFAULT_POSITION  (0, 1, 1);
  const Vector3f PointLight::DEFAULT_ATTENUATION = Vector3f (1, 0, 0);

  PointLight::PointLight (uchar lightNumber) :
    Light (lightNumber),
        m_position (DEFAULT_POSITION), m_attenuation (DEFAULT_ATTENUATION)
  {
  }

  PointLight::~PointLight ()
  {
  }

  void
  PointLight::setPosition (const Vector3f& position)
  {
    m_position = position;
  }

  Vector3f
  PointLight::getPosition () const
  {
    return (m_position);
  }

  void
  PointLight::setConstantAttenuation (Single constantAtten)
  {
    m_attenuation.x = constantAtten;
  }

  Single
  PointLight::getConstantAttenuation () const
  {
    return (m_attenuation.x);
  }

  void
  PointLight::setLinearAttenuation (Single linearAtten)
  {
    m_attenuation.y = linearAtten;
  }

  Single
  PointLight::getLinearAttenuation () const
  {
    return (m_attenuation.y);
  }

  void
  PointLight::setQuadraticAttenuation (Single quadraticAtten)
  {
    m_attenuation.z = quadraticAtten;
  }

  Single
  PointLight::getQuadraticAttenuation () const
  {
    return (m_attenuation.z);
  }

  void
  PointLight::setAttenuation (const Vector3f& attenuation)
  {
    m_attenuation = attenuation;
  }

  Vector3f
  PointLight::getAttenuation () const
  {
    return (m_attenuation);
  }

  Light::LightType
  PointLight::getLightType () const
  {
    return (Light::POINT_LIGHT);
  }
}
