#ifndef MATERIAL_LIBRARY_HPP_
#define MATERIAL_LIBRARY_HPP_

#include <istream>
#include <string>
#include <map>

namespace Exscitech
{
  class Material;
  class MaterialLibrary
  {
  public:
    MaterialLibrary ();

    ~MaterialLibrary ();

    void
    addMaterialsFromFile (const std::string& materialName);

    Exscitech::Material*
    getMaterial (const std::string& materialName) const;

  private:

    void
    readMaterialFile (std::istream& materialFile, const std::string& fileDirectory);

  private:

    std::map<std::string, Material*> m_materialMap;
  };
}

#endif

