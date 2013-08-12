#ifndef BONDS_HPP_
#define BONDS_HPP_

#include <cstdio>
#include "Drawable.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Constants.hpp"

namespace Exscitech
{
  class Bonds : public Drawable
  {
  public:

    Bonds () :
        m_numPoints (0), m_program (
            "./vmd-1.8.7/ExscitechResources/Shaders/CylinderShader.vsh",
            "./vmd-1.8.7/ExscitechResources/Shaders/CylinderShader.fsh",
            "./vmd-1.8.7/ExscitechResources/Shaders/CylinderShader.gsh")
    {
      // Empty. This constructor is used when bonds were not available, to prevent the need for null checking.
    }

    // Unroll for speed!
    Bonds (const std::vector<Vector3f>& points,
        const std::vector<unsigned int>& indices, const Vector4f& details) :
        m_numPoints (indices.size ()), m_details (details), m_program (
            "./vmd-1.8.7/ExscitechResources/Shaders/CylinderShader.vsh",
            "./vmd-1.8.7/ExscitechResources/Shaders/CylinderShader.fsh",
            "./vmd-1.8.7/ExscitechResources/Shaders/CylinderShader.gsh")
    {
      std::vector<Vector3f> unrolledPositions;

      for (unsigned int i = 0; i < indices.size (); ++i)
      {
        unrolledPositions.push_back (points[indices[i]]);
      }
      glGenBuffers (1, &m_vboId);
      glBindBuffer (GL_ARRAY_BUFFER, m_vboId);
      glBufferData (GL_ARRAY_BUFFER, sizeof(Vector3f) * m_numPoints,
          &unrolledPositions[0], GL_STATIC_DRAW);
      glBindBuffer (GL_ARRAY_BUFFER, 0);
    }

    virtual void
    draw (Camera* camera)
    {
      if (m_numPoints > 0)
      {
        m_program.enable ();
        m_program.setUniform (m_program.getUniformLocation ("g_world"),
            getTransform4x4 ());
        m_program.setUniform (m_program.getUniformLocation ("g_view"),
            camera->getView ());
        m_program.setUniform (m_program.getUniformLocation ("g_projection"),
            camera->getProjection ());

        m_program.setUniform (m_program.getUniformLocation ("g_details"),
            m_details);

        m_program.setAttribPointer (m_vboId,
            m_program.getAttribLocation ("g_point1"), 3, GL_FLOAT, GL_FALSE,
            2 * sizeof(Vector3f), 0, 0);
        m_program.setAttribPointer (m_vboId,
            m_program.getAttribLocation ("g_point2"), 3, GL_FLOAT, GL_FALSE,
            2 * sizeof(Vector3f), sizeof(Vector3f), 0);

        glDrawArrays (GL_POINTS, 0, m_numPoints / 2);

        m_program.disableAttribute( m_program.getAttribLocation ("g_point1"));
        m_program.disableAttribute( m_program.getAttribLocation ("g_point2"));
        m_program.disable();

      }
    }

  private:
    unsigned int m_vboId;
    unsigned int m_numPoints;

    Vector4f m_details;
    ShaderProgram m_program;
  };
}
#endif
