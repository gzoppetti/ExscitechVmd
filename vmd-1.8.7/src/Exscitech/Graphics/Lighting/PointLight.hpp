#ifndef POINT_LIGHT_HPP_
#define POINT_LIGHT_HPP_

#include "Exscitech/Graphics/Lighting/Light.hpp"

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"

namespace Exscitech
{
  class PointLight : public Light
  {
    friend Light*
    Light::create (const std::string& name, LightType type);

  protected:

    PointLight (uchar lightNumber);

    virtual
    ~PointLight ();

  public:

    void
    setPosition (const Vector3f& position);

    Vector3f
    getPosition () const;

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
    static const Vector3f DEFAULT_ATTENUATION;

  private:

    Vector3f m_position;
    Vector3f m_attenuation;

  };
}

#endif
