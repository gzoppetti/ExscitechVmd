#include <stdexcept>
#include <iterator>

#include <GL/glew.h>

#include "Exscitech/Graphics/Lighting/Light.hpp"
#include "Exscitech/Graphics/Lighting/PointLight.hpp"
#include "Exscitech/Graphics/Lighting/DirectionalLight.hpp"
#include "Exscitech/Graphics/Lighting/SpotLight.hpp"

namespace Exscitech
{
  const Vector4f Light::DEFAULT_AMBIENT (0, 0, 0, 1);
  const Vector4f Light::DEFAULT_DIFFUSE (1, 1, 1, 1);
  const Vector4f Light::DEFAULT_SPECULAR (1, 1, 1, 1);

  std::bitset<Light::MAX_LIGHTS> Light::ms_lightsUsed;
  Light::LightMap Light::ms_lightMap;

  Light*
  Light::create (const std::string& name, LightType type)
  {
    if (ms_lightMap.size () == MAX_LIGHTS)
    {
      return (NULL);
    }
    uchar i = 0;
    for (; i < MAX_LIGHTS; ++i)
    {
      if (!ms_lightsUsed.test (i))
      {
        break;
      }
    }
    Light* light = NULL;
    if (type == POINT_LIGHT)
    {
      light = new PointLight (i);
    }
    else if (type == DIRECTIONAL_LIGHT)
    {
      light = new DirectionalLight (i);
    }
    else if (type == SPOT_LIGHT)
    {
      light = new SpotLight (i);
    }
    else
    {
      throw std::runtime_error ("Unknown light type");
    }
    ms_lightsUsed.set (i);
    ms_lightMap[name] = light;

    return (light);
  }

  void
  Light::release (const std::string& name)
  {
    LightMap::iterator iter;
    iter = ms_lightMap.find (name);
    if (iter != ms_lightMap.end ())
    {
      // Light was found
      Light* light = iter->second;
      light->disable ();
      ms_lightsUsed.reset (light->m_lightNumber);
      delete light;
      ms_lightMap.erase (iter);
    }
  }

  Light*
  Light::lookup (const std::string& name)
  {
    LightMap::iterator iter;
    iter = ms_lightMap.find (name);
    Light* light = NULL;
    if (iter != ms_lightMap.end ())
    {
      light = iter->second;
    }
    return (light);
  }

  size_t
  Light::getNumLights ()
  {
    return (ms_lightMap.size ());
  }

  std::vector<Light*>
  Light::getLights ()
  {
    std::vector<Light*> lights;
    LightMap::const_iterator citer;
    for (citer = ms_lightMap.begin (); citer != ms_lightMap.end (); ++citer)
    {
      lights.push_back (citer->second);
    }
    return (lights);
  }

  //**

  Light::Light (uchar lightNumber) :
    m_lightNumber (lightNumber), m_ambientColor (DEFAULT_AMBIENT), m_diffuseColor (DEFAULT_DIFFUSE),
    m_specularColor (DEFAULT_SPECULAR)
  {
  }

  Light::~Light ()
  {
  }

  void
  Light::setAmbientColor (const Vector3f& ambient)
  {
    m_ambientColor.set (ambient, 1.0f);
  }

  void
  Light::setAmbientColor (const Vector4f& ambient)
  {
    m_ambientColor = ambient;
  }

  Vector4f
  Light::getAmbientColor () const
  {
    return (m_ambientColor);
  }

  void
  Light::setDiffuseColor (const Vector3f& diffuse)
  {
    m_diffuseColor.set (diffuse, 1.0f);
  }

  void
  Light::setDiffuseColor (const Vector4f& diffuse)
  {
    m_diffuseColor = diffuse;
  }

  Vector4f
  Light::getDiffuseColor () const
  {
    return (m_diffuseColor);
  }

  void
  Light::setSpecularColor (const Vector3f& specular)
  {
    m_specularColor.set (specular, 1.0f);
  }

  void
  Light::setSpecularColor (const Vector4f& specular)
  {
    m_specularColor = specular;
  }

  Vector4f
  Light::getSpecularColor () const
  {
    return (m_specularColor);
  }

  void
  Light::enable ()
  {
    m_isEnabled = true;
  }

  void
  Light::disable ()
  {
    m_isEnabled = false;
  }
}
