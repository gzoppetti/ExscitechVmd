#ifndef NEWMESHPART_HPP_
#define NEWMESHPART_HPP_

#include <string>
#include <vector>

#include <GL/glew.h>

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Graphics/Lighting/Material.hpp"
#include "Exscitech/Graphics/Mesh/MeshPart.hpp"
#include "Exscitech/Graphics/Mesh/Vertex.hpp"
#include "Exscitech/Graphics/Mesh/VertexDescriptor.hpp"
#include "Exscitech/Graphics/Mesh/VertexBuffer.hpp"
#include "Exscitech/Graphics/Mesh/IndexBuffer.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Graphics/Shaders/LightUniformManager.hpp"
#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
  class Mesh;
  class Material;
  class MeshPart : public Drawable
  {
  public:

    MeshPart (const std::string& name, const VertexDescriptor& descriptor, Material* material);

    ~MeshPart();

    uint
    addVertex (const Vertex& vertex);

    void
    addIndices (const std::vector<uint>& indices);

    void
    addIndex (uint index);

    virtual void
    draw (Camera* camera);

    void
    initializeBuffers ();

    VertexDescriptor
    getVertexDescriptor () const;

    void
    bindToMesh (Mesh* boundMesh);

    std::vector<float>&
    getVertexVector ();

    std::vector<uint>&
    getIndexVector ();

    uint
    getVertexSizeInFloats ();

    void
    printBuffers ()
    {
      m_vertexBuffer.printVertices ();
      m_indexBuffer.printIndices ();
    }

  private:

    void
    enableShader(Camera* camera);

    void
    disableShader();

    void
    setUniforms(Camera* camera);

    void
    setAttributes();


  private:

    Material* m_material;
    std::string m_name;
    VertexDescriptor m_vertexDesc;
    VertexBuffer m_vertexBuffer;
    IndexBuffer m_indexBuffer;
    Mesh* m_boundMesh;
    ShaderProgram m_program;
    LightUniformManager m_lightUniformManager;
  };

}

#endif
