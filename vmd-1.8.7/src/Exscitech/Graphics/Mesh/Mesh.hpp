#ifndef MESH_HPP_
#define MESH_HPP_

#include <string>
#include <vector>

#include "Exscitech/Graphics/Drawable.hpp"

#include "Exscitech/Graphics/Mesh/Vertex.hpp"
#include "Exscitech/Graphics/Mesh/VertexDescriptor.hpp"
#include "Exscitech/Graphics/Mesh/VertexBuffer.hpp"
#include "Exscitech/Graphics/Mesh/IndexBuffer.hpp"
#include "Exscitech/Graphics/Mesh/MeshPart.hpp"
#include "Exscitech/Graphics/Lighting/MaterialLibrary.hpp"

namespace Exscitech
{
  class MeshPart;

  class Mesh : public Drawable
  {

  public:

    Mesh (const std::string& name);

    virtual
    ~Mesh ();

    void
    addMeshPart (MeshPart* part);

    MeshPart*
    getMeshPart (uint index);

    void
    initializeMeshParts ();

    void
    draw (Camera* camera);

    void
    printBuffers ();

    uint
    getNumberOfParts ()
    {
      return m_meshParts.size ();
    }

  private:

    std::string m_name;
    std::vector<MeshPart*> m_meshParts;

  };

}

#endif
