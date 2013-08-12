
#include "Exscitech/Graphics/Mesh/VertexAttribute.hpp"

const uchar VertexAttribute::ms_typeSizes[NUM_ATTRIBUTE_TYPES] =
  { 8, 12, 16, 3, 4 };

const uchar VertexAttribute::ms_numberOfComponents[NUM_ATTRIBUTE_TYPES] =
  { 2, 3, 4, 3, 4 };

bool
operator== (const VertexAttribute& attribute1,
    const VertexAttribute& attribute2)
{
  return (attribute1.compare (attribute2) == 0);
}

bool
operator!= (const VertexAttribute& attribute1,
    const VertexAttribute& attribute2)
{
  return (!(attribute1 == attribute2));
}

bool
operator< (const VertexAttribute& attribute1, const VertexAttribute& attribute2)
{
  return (attribute1.compare (attribute2) < 0);
}
