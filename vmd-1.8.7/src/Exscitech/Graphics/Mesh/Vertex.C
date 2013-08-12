
#include "Exscitech/Graphics/Mesh/Vertex.hpp"

namespace Exscitech
{
  Vertex::Vertex () :
    m_descriptor ("DefaultName")
  {
  }

  Vertex::Vertex (const VertexDescriptor& descriptor) :
    m_descriptor (descriptor), m_vertices (descriptor.getVertexSizeInFloats ())
  {
  }

  // Default copy ctor is OK

  Vertex::~Vertex ()
  {
  }

  VertexDescriptor
  Vertex::getDescriptor () const
  {
    return (m_descriptor);
  }

  uchar
  Vertex::getSizeInFloats () const
  {
    return (m_descriptor.getVertexSizeInFloats ());
  }

  void
  Vertex::setAttribute (size_t attributeId, const Single attribValue[])
  {
    uchar offset = m_descriptor.getAttributeOffsetInFloats (attributeId);
    uchar numComponents = m_descriptor.getAttributeNumberOfComponents (
        attributeId);
    for (uchar i = 0; i < numComponents; ++i)
    {
      m_vertices[offset + i] = attribValue[i];
    }
  }

  void
  Vertex::getAttribute (size_t attributeId, Single returnValue[])
  {
    uchar offset = m_descriptor.getAttributeOffsetInFloats (attributeId);
    uchar numComponents = m_descriptor.getAttributeNumberOfComponents (
        attributeId);

    for (uchar i = 0; i < numComponents; ++i)
    {
      returnValue[i] = m_vertices[offset + i];
    }
  }

  bool
  Vertex::compareWithSequence (const float* firstFloat) const
  {
    uchar numFloats = m_descriptor.getVertexSizeInFloats ();
    bool equal = true;
    for (uchar i = 0; i < numFloats; ++i)
    {
      if (!Math<float>::equalsUsingTolerance (m_vertices[i], firstFloat[i]))
      {
        equal = false;
        break;
      }
    }
    return (equal);
  }

  void
  Vertex::writeToBuffer (float* position) const
  {
    for (uchar i = 0; i < m_descriptor.getVertexSizeInFloats (); ++i)
    {
      position[i] = m_vertices[i];
    }
  }

  bool
  Vertex::compareAttributeValue (const Vertex& vertex, size_t attributeId) const
  {
    uchar offset = m_descriptor.getAttributeOffsetInFloats (attributeId);
    uchar numComponents = m_descriptor.getAttributeNumberOfComponents (
        attributeId);
    bool equal = true;
    for (uchar i = 0; i < numComponents; ++i)
    {
      if (!Math<float>::equalsUsingTolerance (m_vertices[offset + i],
          vertex[offset + i]))
      {
        equal = false;
        break;
      }
    }
    return (equal);
  }

  float&
  Vertex::operator[] (size_t index)
  {
    return (m_vertices[index]);
  }

  const float&
  Vertex::operator[] (size_t index) const
  {
    return (m_vertices[index]);
  }

  // Free functions

  bool
  operator== (const Vertex& vertex1, const Vertex& vertex2)
  {
    VertexDescriptor v1Desc = vertex1.getDescriptor ();
    VertexDescriptor v2Desc = vertex2.getDescriptor ();
    bool descEqual = (v1Desc == v2Desc);
    if (!descEqual)
    {
      return (false);
    }
    bool equal = true;
    uchar numFloats = v1Desc.getVertexSizeInFloats ();
    for (size_t i = 0; i < numFloats; ++i)
    {
      if (!Math<float>::equalsUsingTolerance (vertex1[i], vertex2[i]))
      {
        equal = false;
        break;
      }
    }
    return (equal);
  }

  bool
  operator!= (const Vertex& vertex1, const Vertex& vertex2)
  {
    return (!(vertex1 == vertex2));
  }

  bool
  operator< (const Vertex& vertex1, const Vertex& vertex2)
  {
    VertexDescriptor v1Desc = vertex1.getDescriptor ();
    VertexDescriptor v2Desc = vertex2.getDescriptor ();
    bool descEqual = (v1Desc == v2Desc);
    if (!descEqual)
    {
      return (false);
    }

    bool less = false;
    uchar numFloats = v1Desc.getVertexSizeInFloats ();
    for (size_t i = 0; i < numFloats; ++i)
    {
      int compare = Math<float>::compareUsingTolerance (vertex1[i], vertex2[i]);
      if (compare < 0)
      {
        less = true;
        break;
      }
      else if (compare > 0)
      {
        break;
      }
    }
    return (less);
  }

}
