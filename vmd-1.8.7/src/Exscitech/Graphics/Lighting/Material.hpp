#ifndef MATERIAL_HPP_
#define MATERIAL_HPP_

#include <map>
#include <string>

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"

#include "Exscitech/Graphics/Lighting/Texture.hpp"

#include "Exscitech/Types.hpp"

namespace Exscitech
{
  class Texture;

  struct MaterialParameters
  {
    MaterialParameters (const Vector4f& ambient, const Vector4f& diffuse,
        const Vector4f& specular, const Vector4f& emission, float shininess,
        Texture* texture, uint numberOfTiles) :
        m_ambient (ambient), m_diffuse (diffuse), m_specular (specular), m_emission (
            emission), m_shininess (shininess), m_texture (texture), m_numberOfTiles (
            numberOfTiles)
    {
    }

    MaterialParameters () :
        m_ambient (0.2, 0.2, 0.2, 0.2), m_diffuse (0.3, 0.3, 0.3, 0.3), m_specular (
            0.2, 0.2, 0.2, 0.2), m_emission (0.0, 0.0, 0.0, 0.0), m_shininess (
            1), m_texture (new Texture ()), m_numberOfTiles (1)
    {
    }

    void
    setAmbient (const Vector4f& ambient)
    {
      m_ambient = ambient;
    }

    void
    setDiffuse (const Vector4f& diffuse)
    {
      m_diffuse = diffuse;
    }

    void
    setSpecular (const Vector4f& specular)
    {
      m_specular = specular;
    }

    void
    setEmission (const Vector4f& emission)
    {
      m_emission = emission;
    }

    void
    setShininess (float shininess)
    {
      m_shininess = shininess;
    }

    void
    setTexture (Texture* texture)
    {
      m_texture = texture;
    }

    void
    setNumberOfTiles (int tiles)
    {
      m_numberOfTiles = tiles;
    }

    Vector4f m_ambient;
    Vector4f m_diffuse;
    Vector4f m_specular;
    Vector4f m_emission;
    float m_shininess;
    Texture* m_texture;
    uint m_numberOfTiles;
  };

  class Material
  {
  public:

    static Material*
    create (const std::string& name, const MaterialParameters& parameters =
        MaterialParameters ());

    static void
    release (const std::string& name);

    static Material*
    lookup (const std::string& name);

    Material (const MaterialParameters& parameters = MaterialParameters());

    ~Material();

    static void
    clearMap();

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

    void
    setEmissionColor (const Vector3f& emission);

    void
    setEmissionColor (const Vector4f& emission);

    Vector4f
    getEmissionColor () const;

    void
    setShininess (float shininess);

    float
    getShininess () const;

    void
    setTexture (Texture* texture);

    Texture*
    getTexture ();

    void
    setNumberOfTiles (int tiles);

    int
    getNumberOfTiles ();

  private:

    typedef std::map<std::string, Material*> MaterialMap;

  private:

    static MaterialMap ms_materialMap;

  private:

    Vector4f m_ambient;
    Vector4f m_diffuse;
    Vector4f m_specular;
    Vector4f m_emission;
    float m_shininess;
    Texture* m_texture;
    uint m_numberOfTiles;

  };

}

#endif

