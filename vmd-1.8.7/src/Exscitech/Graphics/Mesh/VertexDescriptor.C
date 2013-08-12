#include "Exscitech/Graphics/Mesh/VertexDescriptor.hpp"

namespace Exscitech
{
  bool
  operator== (const VertexDescriptor& desc1, const VertexDescriptor& desc2)
  {
    return (desc1.equals (desc2));
  }

  bool
  operator< (const VertexDescriptor& desc1, const VertexDescriptor& desc2)
  {
    size_t numAttrib1 = desc1.getNumberOfAttributes ();
    size_t numAttrib2 = desc2.getNumberOfAttributes ();
    size_t minAttrib = std::min (numAttrib1, numAttrib2);
    for (size_t i = 0; i < minAttrib; ++i)
    {
      VertexAttribute attrib1 = desc1.getAttribute (i);
      VertexAttribute attrib2 = desc2.getAttribute (i);
      int compare = attrib1.compare (attrib2);
      if (compare < 0)
      {
        return (true);
      }
      else if (compare > 0)
      {
        return (false);
      }
    }
    // Take the one with fewer attributes
    return (numAttrib1 < numAttrib2);
  }
}
