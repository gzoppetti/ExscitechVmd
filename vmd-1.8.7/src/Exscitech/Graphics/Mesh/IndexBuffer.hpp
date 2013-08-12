#ifndef INDEX_BUFFER_HPP_
#define INDEX_BUFFER_HPP_

#include <vector>
#include <cstdio>

#include <GL/glew.h>

#include "Exscitech/Types.hpp"

namespace Exscitech
{
  class IndexBuffer
  {
  public:

    IndexBuffer ();

    ~IndexBuffer ();

    void
    clear ();

    void
    bind () const;

    void
    unbind () const;

    void
    initBuffer ();

    uint
    addIndices (const std::vector<uint>& indices);

    uint
    addIndex (uint index);

    // For debugging purposes
    void
    printIndices ();

    std::vector<uint>&
    getVector ();

    uint
    getNumberOfIndices();

    uint
    getId()
    {
      return m_iboId;
    }
  private:

    GLuint m_iboId;
    std::vector<uint> m_indices;

  };

}

#endif
