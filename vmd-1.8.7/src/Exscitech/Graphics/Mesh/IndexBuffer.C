#include <GL/glew.h>

#include "Exscitech/Graphics/Mesh/IndexBuffer.hpp"

namespace Exscitech
{
  /*PFNGLGENBUFFERSPROC IndexBuffer::ms_glGenBuffers = NULL;
   PFNGLDELETEBUFFERSPROC IndexBuffer::ms_glDeleteBuffers = NULL;
   PFNGLBINDBUFFERPROC IndexBuffer::ms_glBindBuffer = NULL;
   PFNGLBUFFERDATAPROC IndexBuffer::ms_glBufferData = NULL;*/

  IndexBuffer::IndexBuffer ()
  {
//    ms_glGenBuffers = (PFNGLGENBUFFERSPROC)glXGetProcAddress ((const GLubyte*)"glGenBuffers");
//    ms_glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glXGetProcAddress ((const GLubyte*)"glDeleteBuffers");
//    ms_glBindBuffer = (PFNGLBINDBUFFERPROC)glXGetProcAddress ((const GLubyte*)"glBindBuffer");
//    ms_glBufferData = (PFNGLBUFFERDATAPROC)glXGetProcAddress ((const GLubyte*)"glBufferData");
    glGenBuffers (1, &m_iboId);
  }

  IndexBuffer::~IndexBuffer ()
  {
    glDeleteBuffers (1, &m_iboId);
  }

  void
  IndexBuffer::clear ()
  {
    m_indices.clear ();
  }

  void
  IndexBuffer::bind () const
  {
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_iboId);
  }

  void
  IndexBuffer::unbind () const
  {
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);
  }

  void
  IndexBuffer::initBuffer ()
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iboId);
    glBufferData (GL_ELEMENT_ARRAY_BUFFER, sizeof(uint) * m_indices.size (),
        &m_indices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iboId);
  }

  uint
  IndexBuffer::addIndices (const std::vector<uint>& indices)
  {
    // TODO: there should be a better way to do this
    uint indicesAddedLocation = m_indices.size ();
    m_indices.reserve (m_indices.size ());
    for (size_t i = 0; i < indices.size (); ++i)
    {
      m_indices.push_back (indices[i]);
    }
    return (indicesAddedLocation);
  }

  uint
  IndexBuffer::addIndex (uint index)
  {
    m_indices.push_back (index);
    return (m_indices.size () - 1);
  }

  // For debugging purposes
  void
  IndexBuffer::printIndices ()
  {
    for (size_t i = 0; i < m_indices.size (); i += 3)
    {
      fprintf (stderr, "(%u, %u, %u); ", m_indices[i], m_indices[i + 1],
          m_indices[i + 2]);
    }
    fprintf (stderr, "\n");
  }

  std::vector<uint>&
  IndexBuffer::getVector ()
  {
    return m_indices;
  }

  uint
  IndexBuffer::getNumberOfIndices ()
  {
    return m_indices.size ();
  }
}

