#ifndef VERTEXATTRIBUTE_HPP_
#define VERTEXATTRIBUTE_HPP_

#include "Exscitech/Types.hpp"

class VertexAttribute
{
public:

  enum Usage
  {
    POSITION,
    NORMAL,
    TEX_COORD0,
    TEX_COORD1,
    TEX_COORD2,
    COLOR0,
    COLOR1,
    NUM_ATTRIBUTE_USAGES
  };

  enum Type
  {
    FLOAT2, FLOAT3, FLOAT4, UCHAR3, UCHAR4, NUM_ATTRIBUTE_TYPES
  };

public:

  VertexAttribute (Usage usage, Type type) :
    m_usage (usage), m_type (type), m_offsetInBytes (0)
  {
  }

  VertexAttribute (Usage usage, Type type, uchar byteOffset) :
    m_usage (usage), m_type (type), m_offsetInBytes (byteOffset)
  {
  }

  Type
  getType () const
  {
    return (m_type);
  }

  uchar
  getSizeInBytes () const
  {
    return (ms_typeSizes[m_type]);
  }

  uchar
  getNumberOfComponents () const
  {
    return (ms_numberOfComponents[m_type]);
  }

  Usage
  getUsage () const
  {
    return (m_usage);
  }

  uchar
  getOffsetInBytes () const
  {
    return (m_offsetInBytes);
  }

  uchar
  getOffsetInFloats () const
  {
    return (m_offsetInBytes / sizeof(float));
  }

  int
  compare (const VertexAttribute& otherAttrib) const
  {
    int r = m_usage - otherAttrib.m_usage;
    if (r == 0)
    {
      r = m_type - otherAttrib.m_type;
    }
    return (r);
  }

  void
  setByteOffset (uchar byteOffset)
  {
    m_offsetInBytes = byteOffset;
  }

private:

  static const uchar ms_typeSizes[NUM_ATTRIBUTE_TYPES];
  static const uchar ms_numberOfComponents[NUM_ATTRIBUTE_TYPES];

private:

  Usage m_usage;
  Type m_type;
  uchar m_offsetInBytes;

};

bool
operator== (const VertexAttribute& attribute1,
    const VertexAttribute& attribute2);

bool
operator!= (const VertexAttribute& attribute1,
    const VertexAttribute& attribute2);

bool
operator< (const VertexAttribute& attribute1,
    const VertexAttribute& attribute2);

#endif
