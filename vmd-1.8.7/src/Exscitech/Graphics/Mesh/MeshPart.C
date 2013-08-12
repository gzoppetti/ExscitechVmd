#include <vector>
#include <string>
#include <limits>
#include <cstdio>

#include "Exscitech/Graphics/Lighting/Material.hpp"
#include "Exscitech/Graphics/Mesh/MeshPart.hpp"
#include "Exscitech/Graphics/Mesh/Mesh.hpp"
#include "Exscitech/Graphics/Mesh/VertexAttribute.hpp"
#include "Exscitech/Graphics/Mesh/Vertex.hpp"

#include "Exscitech/Utilities/CameraUtility.hpp"
#include "Exscitech/Utilities/DebuggingUtility.hpp"

using std::string;
using std::vector;

namespace Exscitech
{

  MeshPart::MeshPart (const string& name, const VertexDescriptor& descriptor,
      Material* material) :
      m_material (material), m_name (name), m_vertexDesc (descriptor), m_program (
          "./vmd-1.8.7/ExscitechResources/Shaders/GeneralTexturedShader.vsh",
          "./vmd-1.8.7/ExscitechResources/Shaders/GeneralTexturedShader.fsh"), m_lightUniformManager (
          m_program.getId (), 3)
  {
  }

  MeshPart::~MeshPart ()
  {

  }

  uint
  MeshPart::addVertex (const Vertex& vertex)
  {
    uint vertexIndex = m_vertexBuffer.addVertex (vertex);
    return (vertexIndex);
  }

  void
  MeshPart::addIndices (const vector<uint>& indices)
  {
    m_indexBuffer.addIndices (indices);
  }

  void
  MeshPart::addIndex (uint index)
  {
    m_indexBuffer.addIndex (index);
  }

  void
  MeshPart::enableShader (Camera* camera)
  {
    m_program.enable ();

    setUniforms (camera);
    setAttributes ();
  }

  void
  MeshPart::setUniforms (Camera* camera)
  {
    m_lightUniformManager.setUniforms (camera, m_material);
    m_program.setUniform (m_program.getUniformLocation ("g_world"),
        getTransform4x4 ());
    m_program.setUniform (m_program.getUniformLocation ("g_view"),
        camera->getView ());
    m_program.setUniform (m_program.getUniformLocation ("g_projection"),
        camera->getProjection ());
  }

  void
  MeshPart::setAttributes ()
  {
    uint vertexBufferId = m_vertexBuffer.getId ();
    m_indexBuffer.bind ();

    uint vertexSizeInBytes = m_vertexDesc.getVertexSizeInBytes ();
    //fprintf (stderr, "Vertex Size: %i\n", vertexSizeInBytes);

    uint positionOffset = m_vertexDesc.getAttributeOffsetInBytes (
        VertexAttribute::POSITION);

    int positionLocation = m_program.getAttribLocation ("g_position");
    m_program.setAttribPointer (vertexBufferId, positionLocation, 3, GL_FLOAT,
        false, vertexSizeInBytes, positionOffset, 0);

    // fprintf (stderr, "Position Offset: %i\n", positionOffset);

    if (m_vertexDesc.hasAttribute (VertexAttribute::NORMAL))
    {
      uint normalOffset = m_vertexDesc.getAttributeOffsetInBytes (
          VertexAttribute::NORMAL);

      int normalLocation = m_program.getAttribLocation ("g_normal");
      m_program.setAttribPointer (vertexBufferId, normalLocation, 3, GL_FLOAT,
          false, vertexSizeInBytes, normalOffset, 0);

      // fprintf (stderr, "Normal Offset: %i\n", normalOffset);
    }

    if (m_vertexDesc.hasAttribute (VertexAttribute::TEX_COORD0))
    {
      uint texCoordOffset = m_vertexDesc.getAttributeOffsetInBytes (
          VertexAttribute::TEX_COORD0);

      int texPosition = m_program.getAttribLocation ("g_tex0");
      m_program.setAttribPointer (vertexBufferId, texPosition, 2, GL_FLOAT,
          false, vertexSizeInBytes, texCoordOffset, 0);
    }

    if (m_vertexDesc.hasAttribute (VertexAttribute::COLOR0))
    {
      uint initialColorOffset = m_vertexDesc.getAttributeOffsetInBytes (
          VertexAttribute::COLOR0);
      int colorLocation = m_program.getAttribLocation ("g_inColr");
      m_program.setAttribPointer (vertexBufferId, colorLocation, 4, GL_FLOAT,
          false, vertexSizeInBytes, initialColorOffset, 0);
    }
  }

  void
  MeshPart::disableShader ()
  {
    m_vertexBuffer.unbind ();
    m_indexBuffer.unbind ();
    m_program.disableAttribute ("g_position");
    m_program.disableAttribute ("g_normal");
    m_program.disableAttribute ("g_tex0");
    m_program.disableAttribute ("g_inColor");

    m_program.disable ();
  }

  std::vector<float>&
  MeshPart::getVertexVector ()
  {
    return m_vertexBuffer.getVector ();
  }

  std::vector<uint>&
  MeshPart::getIndexVector ()
  {
    return m_indexBuffer.getVector ();
  }

  uint
  MeshPart::getVertexSizeInFloats ()
  {
    return m_vertexDesc.getVertexSizeInFloats ();
  }

  void
  MeshPart::draw (Camera* camera)
  {
    enableShader (camera);

    uint numberOfIndices = m_indexBuffer.getNumberOfIndices ();

    glDrawElements (GL_TRIANGLES, numberOfIndices, GL_UNSIGNED_INT,
        reinterpret_cast<void*> (0));

    disableShader ();
  }

  void
  MeshPart::initializeBuffers ()
  {
    fprintf (stderr, "Generating OpenGL Buffers!\n");
    m_vertexBuffer.initBuffer ();
    m_indexBuffer.initBuffer ();
  }
  VertexDescriptor
  MeshPart::getVertexDescriptor () const
  {
    return (m_vertexDesc);
  }
}

