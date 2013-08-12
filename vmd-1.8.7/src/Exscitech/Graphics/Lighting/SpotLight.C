#include <GL/glew.h>

#include "Exscitech/Graphics/Lighting/SpotLight.hpp"

namespace Exscitech
{
  const Vector3f SpotLight::DEFAULT_POSITION (0, 0, 1);
  const Vector3f SpotLight::DEFAULT_SPOT_DIRECTION (0, 0, -1);
  const Vector3f SpotLight::DEFAULT_ATTENUATION (1, 0, 0);
  const Single SpotLight::DEFAULT_SPOT_EXPONENT = 0;
  const Single SpotLight::DEFAULT_SPOT_CUTOFF = 90;

  SpotLight::SpotLight (uchar lightNumber) :
    Light (lightNumber), m_position (DEFAULT_POSITION)
  {
  }

  SpotLight::~SpotLight ()
  {
  }

  void
  SpotLight::setPosition (const Vector3f& position)
  {
    m_position = position;
  }

  Vector3f
  SpotLight::getPosition () const
  {
    return (m_position);
  }

  void
  SpotLight::setSpotCutoff (Single halfAngle)
  {
    m_spotCutoff = halfAngle;
  }

  Single
  SpotLight::getSpotCutoff () const
  {
    return (m_spotCutoff);
  }

  void
  SpotLight::setSpotExponent (Single exponent)
  {
    glLightf (GL_LIGHT0 + m_lightNumber, GL_SPOT_EXPONENT, exponent);
  }

  Single
  SpotLight::getSpotExponent () const
  {
    Single exponent;
    glGetLightfv (GL_LIGHT0 + m_lightNumber, GL_SPOT_EXPONENT, &exponent);
    return (exponent);
  }

  void
  SpotLight::setSpotDirection (const Vector3f& direction)
  {
    glLightfv (GL_LIGHT0 + m_lightNumber, GL_SPOT_DIRECTION, &direction[0]);
  }

  Vector3f
  SpotLight::getSpotDirection () const
  {
    Vector3f spotDirection;
    glGetLightfv (GL_LIGHT0 + m_lightNumber, GL_SPOT_DIRECTION,
        &spotDirection[0]);
    return (spotDirection);
  }

  void
  SpotLight::setConstantAttenuation (Single constantAtten)
  {
    glLightf (GL_LIGHT0 + m_lightNumber, GL_CONSTANT_ATTENUATION, constantAtten);
  }

  Single
  SpotLight::getConstantAttenuation () const
  {
    Single constantAtten;
    glGetLightfv (GL_LIGHT0 + m_lightNumber, GL_CONSTANT_ATTENUATION,
        &constantAtten);
    return (constantAtten);
  }

  void
  SpotLight::setLinearAttenuation (Single linearAtten)
  {
    glLightf (GL_LIGHT0 + m_lightNumber, GL_LINEAR_ATTENUATION, linearAtten);
  }

  Single
  SpotLight::getLinearAttenuation () const
  {
    Single linearAtten;
    glGetLightfv (GL_LIGHT0 + m_lightNumber, GL_LINEAR_ATTENUATION,
        &linearAtten);
    return (linearAtten);
  }

  void
  SpotLight::setQuadraticAttenuation (Single quadraticAtten)
  {
    glLightf (GL_LIGHT0 + m_lightNumber, GL_QUADRATIC_ATTENUATION,
        quadraticAtten);
  }

  Single
  SpotLight::getQuadraticAttenuation () const
  {
    Single quadraticAtten;
    glGetLightfv (GL_LIGHT0 + m_lightNumber, GL_QUADRATIC_ATTENUATION,
        &quadraticAtten);
    return (quadraticAtten);
  }

  void
  SpotLight::setAttenuation (const Vector3f& attenuation)
  {
    glLightf (GL_LIGHT0 + m_lightNumber, GL_CONSTANT_ATTENUATION,
        attenuation[0]);
    glLightf (GL_LIGHT0 + m_lightNumber, GL_LINEAR_ATTENUATION, attenuation[1]);
    glLightf (GL_LIGHT0 + m_lightNumber, GL_QUADRATIC_ATTENUATION,
        attenuation[2]);
  }

  Vector3f
  SpotLight::getAttenuation () const
  {
    Single constantAtten = getConstantAttenuation ();
    Single linearAtten = getLinearAttenuation ();
    Single quadraticAtten = getQuadraticAttenuation ();
    Vector3f attenuation (constantAtten, linearAtten, quadraticAtten);
    return (attenuation);
  }

  Light::LightType
  SpotLight::getLightType () const
  {
    return (Light::SPOT_LIGHT);
  }
}

