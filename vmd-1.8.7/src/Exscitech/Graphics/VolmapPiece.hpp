#ifndef VOLMAPPIECE_HPP_
#define VOLMAPPIECE_HPP_

#include <vector>

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Graphics/Shaders/LightUniformManager.hpp"
#include "Exscitech/Graphics/Lighting/Material.hpp"

namespace Exscitech
{
  class VolmapPiece : public Drawable
  {
  public:

    VolmapPiece (const std::vector<Vector3f>& vertices,
        const std::vector<Vector3f>& normals) :
        m_program ("./vmd-1.8.7/ExscitechResources/Shaders/GeneralShader.vsh",
            "./vmd-1.8.7/ExscitechResources/Shaders/GeneralShader.fsh"), m_lightUniformManager (
            m_program.getId (), 3)
    {
      fprintf (stderr, "Piece with %lu vertices\n", vertices.size ());

      m_centerOffset.set (0, 0, 0);

      m_vertices.reserve (vertices.size ());

      for (unsigned int i = 0; i < vertices.size (); ++i)
      {
        m_vertices.push_back (vertices[i]);
        m_centerOffset += vertices[i];
      }
      m_centerOffset /= m_vertices.size ();

      m_normals.reserve (normals.size ());
      m_normals.insert (m_normals.begin (), normals.begin (), normals.end ());
    }

    void
    draw (Camera* camera)
    {
      m_program.enable ();
      m_lightUniformManager.setUniforms (camera, &m_material);
      m_program.setUniform (m_program.getUniformLocation ("g_world"),
          getTransform4x4 ());
      m_program.setUniform (m_program.getUniformLocation ("g_view"),
          camera->getView ());
      m_program.setUniform (m_program.getUniformLocation ("g_projection"),
          camera->getProjection ());

      m_program.setAttribPointer (m_program.getAttribLocation ("g_position"), 3,
          GL_FLOAT, false, 0, 0, &m_vertices[0]);
      m_program.setAttribPointer (m_program.getAttribLocation ("g_normal"), 3,
          GL_FLOAT, false, 0, 0, &m_normals[0]);

      glDrawArrays (GL_TRIANGLES, 0, m_vertices.size ());

      m_program.disableAttribute ("g_position");
      m_program.disableAttribute ("g_normal");
      m_program.disable ();

    }

    Vector3f
    getOffset ()
    {
      return m_centerOffset;
    }

  private:

    std::vector<Vector3f> m_vertices;
    std::vector<Vector3f> m_normals;
    Vector3f m_centerOffset;
    ShaderProgram m_program;
    Material m_material;
    LightUniformManager m_lightUniformManager;

  };
}
#endif
