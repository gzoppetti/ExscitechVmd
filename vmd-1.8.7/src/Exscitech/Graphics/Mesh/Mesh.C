#include <cstdio>

#include "Exscitech/Graphics/Mesh/Mesh.hpp"
#include "Exscitech/Graphics/Mesh/MeshPart.hpp"

namespace Exscitech
{
  using std::string;
  using std::vector;

  Mesh::Mesh (const std::string& name) :
      m_name (name)
  {
    //setShaderParameters (new GeneralPurposeShaderParameters ());
  }

  Mesh::~Mesh ()
  {
    for (size_t i = 0; i < m_meshParts.size (); ++i)
    {
      delete m_meshParts[i];
    }
  }

  void
  Mesh::addMeshPart (MeshPart* part)
  {
    m_meshParts.push_back (part);
  }

  MeshPart*
  Mesh::getMeshPart (uint index)
  {
    MeshPart* part = NULL;

    if (index < m_meshParts.size ())
    {
      part = m_meshParts[index];
    }
    return part;
  }

  void
  Mesh::initializeMeshParts ()
  {
    for (size_t i = 0; i < m_meshParts.size (); ++i)
    {
      m_meshParts[i]->initializeBuffers ();
    }
  }
  void
  Mesh::draw (Camera* camera)
  {
    for (size_t i = 0; i < m_meshParts.size (); ++i)
    {
      m_meshParts[i]->setTransform(getTransform());
      m_meshParts[i]->draw (camera);
    }
  }

  void
  Mesh::printBuffers ()
  {
    for (size_t i = 0; i < m_meshParts.size (); ++i)
    {
      m_meshParts[i]->printBuffers ();
    }
  }

}
