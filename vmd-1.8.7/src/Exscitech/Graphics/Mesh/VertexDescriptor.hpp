#ifndef VERTEXDESCRIPTOR_HPP_
#define VERTEXDESCRIPTOR_HPP_

#include <string>
#include <vector>
#include <climits>

#include "Exscitech/Graphics/Mesh/VertexAttribute.hpp"

namespace Exscitech
{
  class VertexDescriptor
  {
  public:

    VertexDescriptor () :
        m_name (""), m_sizeInBytes (0)
    {
    }

    VertexDescriptor (const std::string& name) :
        m_name (name), m_sizeInBytes (0)
    {
    }

    void
    setName (const std::string& name)
    {
      m_name = name;
    }

    // Return an attribute ID
    size_t
    addAttribute (VertexAttribute::Usage usage, VertexAttribute::Type type)
    {
      VertexAttribute attribute (usage, type, m_sizeInBytes);
      m_sizeInBytes += attribute.getSizeInBytes ();
      size_t attributeId = m_attributes.size ();
      m_attributes.push_back (attribute);

      return (attributeId);
    }

    const VertexAttribute&
    getAttribute (size_t index) const
    {
      return (m_attributes[index]);
    }

    size_t
    getNumberOfAttributes () const
    {
      return (m_attributes.size ());
    }

    uchar
    getAttributeNumberOfComponents (size_t attributeId) const
    {
      return (m_attributes[attributeId].getNumberOfComponents ());
    }

    uchar
    getVertexSizeInBytes () const
    {
      return (m_sizeInBytes);
    }

    uchar
    getVertexSizeInFloats () const
    {
      return (getVertexSizeInBytes () / sizeof(float));
    }

    uchar
    getAttributeOffsetInFloats (size_t attributeId) const
    {
      uchar offsetInBytes = getAttributeOffsetInBytes (attributeId);
      return (offsetInBytes / sizeof(float));
    }

    uchar
    getAttributeOffsetInBytes (size_t attributeId) const
    {
      return (m_attributes[attributeId].getOffsetInBytes ());
    }

    uchar
    getAttributeOffsetInBytes (VertexAttribute::Usage usage) const
    {
      uchar offset = UCHAR_MAX;
      for (size_t i = 0; i < m_attributes.size (); ++i)
      {
        const VertexAttribute& attribute = m_attributes[i];
        if (attribute.getUsage () == usage)
        {
          offset = attribute.getOffsetInBytes ();
          break;
        }
      }
      return (offset);
    }

    bool
    hasAttribute (VertexAttribute::Usage usage) const
    {
      bool found = false;
      for (size_t i = 0; i < m_attributes.size (); ++i)
      {
        const VertexAttribute& attribute = m_attributes[i];
        if (attribute.getUsage () == usage)
        {
          found = true;
          break;
        }
      }
      return (found);
    }

    size_t
    getAttributeIdByUsage (VertexAttribute::Usage usage) const
    {
      size_t found = UINT_MAX;
      for (size_t i = 0; i < m_attributes.size (); ++i)
      {
        const VertexAttribute& attribute = m_attributes[i];
        if (attribute.getUsage () == usage)
        {
          found = i;
          break;
        }
      }
      return (found);
    }

    bool
    equals (const VertexDescriptor& otherDescriptor) const
    {
      return (m_attributes == otherDescriptor.m_attributes);
    }

  private:

    std::string m_name;
    uchar m_sizeInBytes;
    std::vector<VertexAttribute> m_attributes;

  };

  bool
  operator== (const VertexDescriptor& desc1, const VertexDescriptor& desc2);

  bool
  operator< (const VertexDescriptor& desc1, const VertexDescriptor& desc2);

}

#endif
