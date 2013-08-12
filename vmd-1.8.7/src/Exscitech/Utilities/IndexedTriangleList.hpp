#ifndef INDEXED_TRIANGLE_LIST_HPP_
#define INDEXED_TRIANGLE_LIST_HPP_

#include <iterator>
#include <algorithm>

#include "Exscitech/Types.hpp"
#include "Exscitech/Graphics/Mesh/Vertex.hpp"

namespace Exscitech
{
  class PositionCollection
  {
  public:

    float*
    getBasePointer ()
    {
      return (&m_positions[0][0]);
    }

    uint
    getStrideInBytes () const
    {
      return (3 * sizeof(float));
    }

    size_t
    size () const
    {
      return (m_positions.size ());
    }

    void
    addPosition (const Vector3f& position)
    {
      m_positions.push_back (position);
    }

  private:

    std::vector<Vector3f> m_positions;

  };

  class IndexCollection
  {
  public:

    uint*
    getBasePointer ()
    {
      return (&m_indices[0]);
    }

    uint
    getStrideInBytes () const
    {
      return (sizeof(uint));
    }

    size_t
    size () const
    {
      return (m_indices.size ());
    }

    void
    addTriangleIndices (uint index1, uint index2, uint index3)
    {
      m_indices.push_back (index1);
      m_indices.push_back (index2);
      m_indices.push_back (index3);
    }

    void
    addIndices (const uint indices[], size_t numIndices)
    {
      std::copy (indices, indices + numIndices, std::back_inserter (m_indices));
    }

  private:

    std::vector<uint> m_indices;

  };

  class IndexedTriangleList
  {
  public:

    float*
    getVertexBasePtr ()
    {
      return (m_positions.getBasePointer ());
    }

    uint*
    getIndexBasePtr ()
    {
      return (m_indices.getBasePointer ());
    }

    uint
    getVertexStrideInBytes () const
    {
      return (m_positions.getStrideInBytes ());
    }

    uint
    getIndexStrideInBytes () const
    {
      return (m_indices.getStrideInBytes ());
    }

    size_t
    getNumTriangles () const
    {
      return (m_indices.size () / 3);
    }

    size_t
    getNumVertices () const
    {
      return (m_positions.size ());
    }

    void
    addPosition (const Vector3f& position)
    {
      m_positions.addPosition (position);
    }

    void
    addPosition (float x, float y, float z)
    {
      addPosition (Vector3f (x, y, z));
    }

    void
    addTriangle (uint index1, uint index2, uint index3)
    {
      m_indices.addTriangleIndices (index1, index2, index3);
    }

    void
    addIndices (const uint indices[], size_t numIndices)
    {
      m_indices.addIndices (indices, numIndices);
    }

  private:

    PositionCollection m_positions;
    IndexCollection m_indices;

  };
}

#endif 
