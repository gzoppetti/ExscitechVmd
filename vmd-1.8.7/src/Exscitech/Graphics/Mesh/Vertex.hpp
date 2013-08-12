#ifndef VERTEX_HPP_
#define VERTEX_HPP_

#include <vector>

#include "Exscitech/Graphics/Mesh/VertexDescriptor.hpp"

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Math.hpp"

namespace Exscitech
{
  class Vertex
  {
  public:

    Vertex ();

    explicit
    Vertex (const VertexDescriptor& descriptor);

    // Default copy ctor is OK

    ~Vertex ();

    VertexDescriptor
    getDescriptor () const;

    uchar
    getSizeInFloats () const;

    void
    setAttribute (size_t attributeId, const Single attribValue[]);

    void
    getAttribute (size_t attributeId, Single returnValue[]);

    bool
    compareWithSequence (const float* firstFloat) const;

    void
    writeToBuffer (float* position) const;

    bool
    compareAttributeValue (const Vertex& vertex, size_t attributeId) const;

    float&
    operator[] (size_t index);

    const float&
    operator[] (size_t index) const;

  private:

    VertexDescriptor m_descriptor;
    std::vector<float> m_vertices;

  };

  bool
  operator== (const Vertex& vertex1, const Vertex& vertex2);

  bool
  operator!= (const Vertex& vertex1, const Vertex& vertex2);

  bool
  operator< (const Vertex& vertex1, const Vertex& vertex2);

}

#endif
