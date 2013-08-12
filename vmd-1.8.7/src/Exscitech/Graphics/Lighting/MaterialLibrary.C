#include <sstream>
#include <string>
#include <istream>
#include <fstream>
#include <iostream>
#include <map>

#include <boost/filesystem.hpp>

#include "Exscitech/Graphics/Lighting/Material.hpp"
#include "Exscitech/Graphics/Lighting/MaterialLibrary.hpp"
#include "Exscitech/Graphics/Lighting/Texture.hpp"
#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
  typedef std::map<std::string, Material*>::const_iterator MaterialConstIterator;

  MaterialLibrary::MaterialLibrary ()
  {
    m_materialMap["default"] = Material::create ("default");
  }

  MaterialLibrary::~MaterialLibrary ()
  {
    for (MaterialConstIterator i = m_materialMap.begin ();
        i != m_materialMap.end (); ++i)
        {
      delete i->second;
    }
  }

  void
  MaterialLibrary::addMaterialsFromFile (const std::string& file)
  {
    namespace fs = boost::filesystem;
    fs::path path (file);
    fs::path directory = path.parent_path ();
    std::string fileDirectory = directory.string () + "/";
    fprintf (stderr, "%s\n", fileDirectory.c_str ());

    std::string nextFileName;

    std::ifstream materialFileIn;
    materialFileIn.open (file.c_str ());
    if (!materialFileIn)
    {
      fprintf (stderr, "Error: could not open material file.\n");
    }
    else
    {
      readMaterialFile (materialFileIn, fileDirectory);
    }
    materialFileIn.close ();
  }

  Exscitech::Material*
  MaterialLibrary::getMaterial (const std::string& materialName) const
  {
    MaterialConstIterator matLoc = m_materialMap.find (materialName);
    if (matLoc == m_materialMap.end ())
    {
      matLoc = m_materialMap.find ("default");
    }
    return (matLoc->second);
  }

  void
  MaterialLibrary::readMaterialFile (std::istream& materialFile,
      const std::string& fileDirectory)
  {
    Exscitech::Material* material = NULL;
    std::string line;
    while (std::getline (materialFile, line))
    {
      std::stringstream lineStream (line);
      std::string token;
      lineStream >> token;

      if (token == "newmtl")
      {
        std::string materialName;
        lineStream >> materialName;
        material = Material::create (materialName);
        if (material == NULL)
        {
          material = Material::lookup (materialName);
        }
        m_materialMap.insert (std::make_pair (materialName, material));
      }
      else if (token == "Ka")
      {
        Vector3f color;
        lineStream >> color.x >> color.y >> color.z;
        material->setAmbientColor (color);
      }
      else if (token == "Kd")
      {
        Vector3f color;
        lineStream >> color.x >> color.y >> color.z;
        material->setDiffuseColor (color);
      }
      else if (token == "Ks")
      {
        Vector3f color;
        lineStream >> color.x >> color.y >> color.z;
        material->setSpecularColor (color);
      }
      else if (token == "illum")
      {
        // Illumination model -- not handling
      }
      else if (token == "Ns")
      {
        float shininess;
        lineStream >> shininess;
        material->setShininess (shininess);
      }
      else if (token == "map_Kd")
      {
        std::string textureFile;
        lineStream >> textureFile;
        std::string textureFileWithDirectory = fileDirectory + textureFile;
        fprintf (stderr, "%s\n", textureFileWithDirectory.c_str ());
        Texture* texture = Texture::create (textureFile,
            textureFileWithDirectory);

        material->setTexture (texture);
        m_materialMap.insert (make_pair (textureFile, material));
      }
    }
  }

}
