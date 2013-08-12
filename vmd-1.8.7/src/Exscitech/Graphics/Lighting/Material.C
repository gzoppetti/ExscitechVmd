#include <GL/glew.h>

#include <sstream>

#include "Exscitech/Graphics/Lighting/Texture.hpp"
#include "Exscitech/Graphics/Lighting/Material.hpp"

namespace Exscitech
{
  Material::MaterialMap Material::ms_materialMap;

  Material*
  Material::create (const std::string& name,
      const MaterialParameters& parameters)
  {
    MaterialMap::iterator iter;
    Material* material = NULL;

    iter = ms_materialMap.find (name);
    if (iter == ms_materialMap.end ())
    {
      material = new Material (parameters);
      ms_materialMap[name] = material;
      return material;
    }
    else
    {
      return iter->second;
    }
  }

  void
  Material::release (const std::string& name)
  {
    MaterialMap::iterator iter;
    iter = ms_materialMap.find (name);
    if (iter != ms_materialMap.end ())
    {
      // Material was found
      Material* material = iter->second;
      delete material;
      ms_materialMap.erase (iter);
    }
  }

  Material*
  Material::lookup (const std::string& name)
  {
    MaterialMap::iterator iter;
    iter = ms_materialMap.find (name);
    Material* material = NULL;
    if (iter != ms_materialMap.end ())
    {
      material = iter->second;
    }
    return material;
  }

  void
  Material::clearMap ()
  {
    for (MaterialMap::iterator iter = ms_materialMap.begin ();
        iter != ms_materialMap.end (); ++iter)
    {
      // Y U NO LIKE DELETE?
      //delete iter->second;
    }
    ms_materialMap.clear();
  }

  Material::Material (const MaterialParameters& parameters) :
      m_ambient (parameters.m_ambient), m_diffuse (parameters.m_diffuse), m_specular (
          parameters.m_specular), m_emission (parameters.m_emission), m_shininess (
          parameters.m_shininess), m_texture (parameters.m_texture), m_numberOfTiles (
          parameters.m_numberOfTiles)
  {
    std::string name;
    do
    {
      std::stringstream ss;
      ss << rand ();
      name = ss.str ();
    }
    while (Material::lookup (name) != NULL);

    ms_materialMap[name] = this;
  }

  Material::~Material()
  {

  }

  void
  Material::setAmbientColor (const Vector3f& ambient)
  {
    m_ambient.set (ambient, 1.0f);
  }

  void
  Material::setAmbientColor (const Vector4f& ambient)
  {
    m_ambient = ambient;
  }

  Vector4f
  Material::getAmbientColor () const
  {
    return m_ambient;
  }

  void
  Material::setDiffuseColor (const Vector3f& diffuse)
  {
    m_diffuse.set (diffuse, 1.0f);
  }

  void
  Material::setDiffuseColor (const Vector4f& diffuse)
  {
    m_diffuse = diffuse;
  }

  Vector4f
  Material::getDiffuseColor () const
  {
    return m_diffuse;
  }

  void
  Material::setSpecularColor (const Vector3f& specular)
  {
    m_specular.set (specular, 1.0f);
  }

  void
  Material::setSpecularColor (const Vector4f& specular)
  {
    m_specular = specular;
  }

  Vector4f
  Material::getSpecularColor () const
  {
    return m_specular;
  }

  void
  Material::setEmissionColor (const Vector3f& emission)
  {
    m_emission.set (emission, 1.0f);
  }

  void
  Material::setEmissionColor (const Vector4f& emission)
  {
    m_emission = emission;
  }

  Vector4f
  Material::getEmissionColor () const
  {
    return m_emission;
  }

  void
  Material::setShininess (float shininess)
  {
    m_shininess = shininess;
  }

  float
  Material::getShininess () const
  {
    return m_shininess;
  }

  void
  Material::setTexture (Texture* texture)
  {
    m_texture = texture;
  }

  Texture*
  Material::getTexture ()
  {
    return m_texture;
  }

  void
  Material::setNumberOfTiles (int tiles)
  {
    m_numberOfTiles = tiles;
  }

  int
  Material::getNumberOfTiles ()
  {
    return m_numberOfTiles;
  }
}

