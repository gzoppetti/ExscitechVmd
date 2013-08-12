#include "Exscitech/Graphics/FullQuad.hpp"

namespace Exscitech
{
  const float FullQuad::VERTS[18] =
    { -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };
  FullQuad::FullQuad () :
      m_texId (0), m_program ("./vmd-1.8.7/ExscitechResources/Shaders/FullQuadShader.vsh",
          "./vmd-1.8.7/ExscitechResources/Shaders/FullQuadShader.fsh")
  {
    glGenBuffers (1, &m_vboId);
    glBindBuffer (GL_ARRAY_BUFFER, m_vboId);
    glBufferData (GL_ARRAY_BUFFER, sizeof(float) * 18, VERTS, GL_STATIC_DRAW);

    glBindBuffer (GL_ARRAY_BUFFER, 0);
  }

  FullQuad::~FullQuad ()
  {
    glDeleteBuffers (1, &m_vboId);
  }

  void
  FullQuad::setTexture (unsigned int texId)
  {
    m_texId = texId;
  }

  void
  FullQuad::draw (int texUnit)
  {
    glActiveTexture (GL_TEXTURE0 + texUnit);
    glBindTexture (GL_TEXTURE_2D, m_texId);
    glEnable (GL_TEXTURE_2D);
    m_program.enable ();
    m_program.setUniform (m_program.getUniformLocation ("g_sampler"), texUnit);

    m_program.setAttribPointer (m_vboId,
        m_program.getAttribLocation ("g_position"), 3, GL_FLOAT, GL_FALSE,
        3 * sizeof(float), 0, 0);

    glDrawArrays (GL_TRIANGLES, 0, 6);

    m_program.disable ();
  }
}
