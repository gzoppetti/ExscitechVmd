
#include <string>
#include <istream>
#include <fstream>
#include <iostream>
#include <map>

#include "Exscitech/Graphics/Mesh/MaterialLibrary.hpp"

using std::istream;
using std::ifstream;
using std::string;

typedef std::map<string, Material*>::const_iterator MaterialConstIterator;

MaterialLibrary::MaterialLibrary ()
{
  m_materialIndex["default"] = new Material ();
}

MaterialLibrary::~MaterialLibrary ()
{
  for (MaterialConstIterator i = m_materialIndex.begin ();
       i != m_materialIndex.end (); ++i)
  {
    delete i->second;
  }
}

void
MaterialLibrary::addMaterialsFromFile (istream& input)
{
  string nextFileName;
  ifstream materialFileIn;
  while (input >> nextFileName)
  {
    materialFileIn.open (nextFileName.c_str ());
    if (!materialFileIn)
    {
      fprintf (stderr, "Error: could not open material file.\n");
    }
    else
    {
      readMaterialFile (materialFileIn);
    }
    materialFileIn.close ();
  }
}

Material*
MaterialLibrary::getMaterial (string materialName) const
{
  MaterialConstIterator matLoc = m_materialIndex.find (materialName);
  if (matLoc == m_materialIndex.end ())
  {
    matLoc = m_materialIndex.find ("default");
  }
  return (matLoc->second);
}

void
MaterialLibrary::readMaterialFile (istream& materialFile)
{
  // TODO implement
}

