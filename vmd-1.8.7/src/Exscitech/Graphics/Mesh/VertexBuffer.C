#include <cstdio>

#include "Exscitech/Graphics/Mesh/VertexBuffer.hpp"

namespace Exscitech
{
  VertexBuffer::VertexBuffer () :
      m_insertionPoint (0), m_baseInsertionPoint (0), m_lastVertexSize (1)
  {
    glGenBuffers (1, &m_vboId);
  }

  VertexBuffer::~VertexBuffer ()
  {
    glDeleteBuffers (1, &m_vboId);
  }

  void
  VertexBuffer::clear ()
  {
    m_insertionPoint = 0;
    m_vertices.clear ();
  }

  void
  VertexBuffer::bind () const
  {
    glBindBuffer (GL_ARRAY_BUFFER, m_vboId);
  }

  void
  VertexBuffer::unbind () const
  {
    glBindBuffer (GL_ARRAY_BUFFER, 0);
  }

  void
  VertexBuffer::initBuffer ()
  {
    glBindBuffer (GL_ARRAY_BUFFER, m_vboId);
    glBufferData (GL_ARRAY_BUFFER, sizeof(float) * m_vertices.size (),
        &m_vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  uint
  VertexBuffer::getId () const
  {
    return m_vboId;
  }

  uint
  VertexBuffer::getInitialAttributeOffset (uint partStartingPos) const
  {
    //Couldnt this just be partStartingPos * sizeof(float)?

    uint distanceInFloats = (&m_vertices[partStartingPos] - &m_vertices[0]);
    return (distanceInFloats * sizeof(float));
  }

  uint
  VertexBuffer::addVertex (const Vertex& vertex)
  {
    uchar vertexSizeInFloats = vertex.getSizeInFloats ();

    for (size_t i = 0; i < vertexSizeInFloats; ++i)
    {
      m_vertices.push_back (vertex[i]);
    }

    uint insertionPoint = m_insertionPoint;
    m_insertionPoint += vertexSizeInFloats;

    uint vertexId = insertionPoint / vertexSizeInFloats;
    return (vertexId);
  }

  uint
  VertexBuffer::addToVertexMap (const Vertex& vertex)
  {
//    typedef boost::bimap<Vertex, uint> VertexMap;
//    typedef VertexMap::left_map::const_iterator VertexMapConstIter;
//    // Use m_insertionPoint as a vertex index
//    uint vertexId = m_insertionPoint;
//    std::pair<VertexMapConstIter, bool> p = m_vertexMap.left.insert (
//        std::make_pair (vertex, vertexId));
//    if (!p.second)
//    {
//      // Already present, so use existing ID
//      vertexId = p.first->second;
//    }
//    else
//    {
//      // Insert succeeded
//      ++m_insertionPoint;
//    }
//    return (vertexId);
  }

  void
  VertexBuffer::copyMapToBuffer ()
  {
//    typedef boost::bimap<Vertex, uint> VertexMap;
//    typedef VertexMap::right_map::const_iterator VertexMapConstIter;
//    VertexMapConstIter i = m_vertexMap.right.begin ();
//    uint sizeOfVertex = i->second.getSizeInFloats ();
//    uint size = m_vertexMap.size () * sizeOfVertex;
//    m_vertices.resize (size);
//    float* floatPointer = &m_vertices[0];
//    for (; i != m_vertexMap.right.end (); ++i)
//    {
//      i->second.writeToBuffer (floatPointer);
//      floatPointer += sizeOfVertex;
//    }
//    m_vertexMap.clear ();
  }

  uint
  VertexBuffer::findVertex (const Vertex& vertex, uint startIndex) const
  {
    uchar vertexSizeInFloats = vertex.getSizeInFloats ();
    uint foundAt = UINT_MAX;
    for (uint i = startIndex; i < m_insertionPoint; i += vertexSizeInFloats)
    {
      if (vertex.compareWithSequence (&m_vertices[i]))
      {
        foundAt = i / vertexSizeInFloats;
        break;
      }
    }

    return (foundAt);
  }

  // For debugging purposes
  void
  VertexBuffer::printVertices ()
  {
    const uint floatsPerVertex = 8;
    std::cerr << "Vertices: " << m_vertices.size () / floatsPerVertex
        << std::endl;
    for (uint i = 0; i < m_vertices.size (); i += floatsPerVertex)
    {
      fprintf (stderr, "Pos: %.2f, %.2f, %.2f \n", m_vertices[i],
          m_vertices[i + 1], m_vertices[i + 2]);

      fprintf (stderr, "Nor: %.2f, %.2f, %.2f \n", m_vertices[i + 3],
          m_vertices[i + 4], m_vertices[i + 5]);

      fprintf (stderr, "Tex: %.2f, %.2f \n", m_vertices[i + 6],
          m_vertices[i + 7]);
    }
    fprintf (stderr, "\n");
  }

  std::vector<float>&
  VertexBuffer::getVector ()
  {
    return m_vertices;
  }
}
