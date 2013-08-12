
#ifndef MATERIALLIBRARY_HPP_
#define MATERIALLIBRARY_HPP_

#include <istream>
#include <string>
#include <map>

#include "Exscitech/Graphics/Lighting/Material.hpp"

class MaterialLibrary
{
public:
  MaterialLibrary ();

  ~MaterialLibrary ();

  void
  addMaterialsFromFile (std::istream& input);

  Material*
  getMaterial (const std::string materialName) const;

private:

  void
  readMaterialFile (std::istream& materialFile);

private:

  std::map<std::string, Material*> m_materialIndex;
};

#endif

