#ifndef VERTEXBUFFER_HPP_
#define VERTEXBUFFER_HPP_

#include <vector>

#include <GL/glew.h>

//#include <boost/bimap.hpp>

#include "Exscitech/Types.hpp"

#include "Exscitech/Graphics/Mesh/Vertex.hpp"

namespace Exscitech
{
  class VertexBuffer
  {
  public:

    VertexBuffer ();

    ~VertexBuffer ();

    void
    clear ();

    void
    bind () const;

    void
    unbind () const;

    void
    initBuffer ();

    uint
    getId () const;

    uint
    getInitialAttributeOffset (uint partStartingPos) const;

    uint
    addVertex (const Vertex& vertex);

    uint
    addToVertexMap (const Vertex& vertex);

    void
    copyMapToBuffer ();

    uint
    findVertex (const Vertex& vertex, uint startIndex) const;

    // For debugging purposes
    void
    printVertices ();

    std::vector<float>&
    getVector();

    uint
    getNumberOfVertices(float vertexSize)
    {
      return m_vertices.size() / vertexSize;
    }
  private:

    GLuint m_vboId;
    uint m_insertionPoint;
    uint m_baseVertexIndex;
    uint m_baseInsertionPoint;

    uint m_lastVertexSize;
    std::vector<float> m_vertices;
    //boost::bimap<Vertex, uint> m_vertexMap;

  };

}

#endif
