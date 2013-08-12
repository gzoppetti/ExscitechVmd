#ifndef DIRECTIONAL_LIGHT_HPP_
#define DIRECTIONAL_LIGHT_HPP_

#include "Exscitech/Graphics/Lighting/Light.hpp"
#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"

namespace Exscitech
{
  class DirectionalLight : public Light
  {
    friend Light*
    Light::create (const std::string& name, LightType type);

  protected:

    DirectionalLight (uchar lightNumber);

    virtual
    ~DirectionalLight ();

  public:

    void
    setDirection (const Vector3f& direction);

    Vector3f
    getDirection () const;

    virtual LightType
    getLightType () const;

  public:

    static const Vector3f DEFAULT_DIRECTION;

  private:

    Vector3f m_direction;

  };
}

#endif
