#ifndef LIGHT_HPP
#define LIGHT_HPP

#include <bitset>
#include <string>
#include <map>
#include <vector>

#include <GL/glew.h>

#include "Exscitech/Types.hpp"

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"

namespace Exscitech
{
  class Light
  {
  public:

    enum LightType
    {
      DIRECTIONAL_LIGHT, POINT_LIGHT, SPOT_LIGHT
    };

  public:

    static Light*
    create (const std::string& name, LightType type);

    static void
    release (const std::string& name);

    static Light*
    lookup (const std::string& name);

  protected:

    Light (uchar lightNumber);

    virtual
    ~Light ();

  public:

    static size_t
    getNumLights ();

    static std::vector<Light*>
    getLights ();

  public:

    void
    setAmbientColor (const Vector3f& ambient);

    void
    setAmbientColor (const Vector4f& ambient);

    Vector4f
    getAmbientColor () const;

    void
    setDiffuseColor (const Vector3f& diffuse);

    void
    setDiffuseColor (const Vector4f& diffuse);

    Vector4f
    getDiffuseColor () const;

    void
    setSpecularColor (const Vector3f& specular);

    void
    setSpecularColor (const Vector4f& specular);

    Vector4f
    getSpecularColor () const;

    virtual LightType
    getLightType () const = 0;

    void
    enable ();

    void
    disable ();

  public:

    static const uchar MAX_LIGHTS = 8;
    static const Vector4f DEFAULT_AMBIENT;
    static const Vector4f DEFAULT_DIFFUSE;
    static const Vector4f DEFAULT_SPECULAR;

  private:

    typedef std::map<std::string, Light*> LightMap;
    static std::bitset<MAX_LIGHTS> ms_lightsUsed;
    static LightMap ms_lightMap;

  protected:

    uchar m_lightNumber;
    Vector4f m_ambientColor;
    Vector4f m_diffuseColor;
    Vector4f m_specularColor;

    bool m_isEnabled;

  };
}

#endif

