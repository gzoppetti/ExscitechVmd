#ifndef SPOT_LIGHT_HPP_
#define SPOT_LIGHT_HPP_

#include "Exscitech/Graphics/Lighting/Light.hpp"

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"

namespace Exscitech
{
  class SpotLight : public Light
  {
    friend Light*
    Light::create (const std::string& name, LightType type);

  protected:

    SpotLight (uchar lightNumber);

    virtual
    ~SpotLight ();

  public:

    void
    setPosition (const Vector3f& position);

    Vector3f
    getPosition () const;

    void
    setSpotCutoff (Single halfAngle);

    Single
    getSpotCutoff () const;

    void
    setSpotExponent (Single exponent);

    Single
    getSpotExponent () const;

    void
    setSpotDirection (const Vector3f& direction);

    Vector3f
    getSpotDirection () const;

    void
    setConstantAttenuation (Single constantAtten);

    Single
    getConstantAttenuation () const;

    void
    setLinearAttenuation (Single linearAtten);

    Single
    getLinearAttenuation () const;

    void
    setQuadraticAttenuation (Single quadraticAtten);

    Single
    getQuadraticAttenuation () const;

    void
    setAttenuation (const Vector3f& attenuation);

    Vector3f
    getAttenuation () const;

    virtual LightType
    getLightType () const;

  public:

    static const Vector3f DEFAULT_POSITION;

    static const Vector3f DEFAULT_SPOT_DIRECTION;
    static const Vector3f DEFAULT_ATTENUATION;
    static const Single DEFAULT_SPOT_EXPONENT;
    static const Single DEFAULT_SPOT_CUTOFF;

  private:

    Vector3f m_position;
    float m_spotCutoff;

  };
}

#endif
