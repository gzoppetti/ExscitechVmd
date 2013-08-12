#include <GL/glew.h>

#include <vector>
#include <string>

#include "Exscitech/Graphics/Skybox.hpp"
#include "Exscitech/Graphics/Lighting/Texture.hpp"
#include "Exscitech/Graphics/Lighting/Material.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"

namespace Exscitech
{

#define POS 0.5f
#define NEG -0.5f
  // Y -Y, X, -X, Z, -Z
  const Vector3f Skybox::ms_vertices[36] =
    {
        Vector3f (NEG, POS, NEG),
        Vector3f (NEG, POS, POS),
        Vector3f (POS, POS, POS),
        Vector3f (NEG, POS, NEG),
        Vector3f (POS, POS, POS),
        Vector3f (POS, POS, NEG),
        Vector3f (NEG, NEG, POS),
        Vector3f (NEG, NEG, NEG),
        Vector3f (POS, NEG, NEG),
        Vector3f (NEG, NEG, POS),
        Vector3f (POS, NEG, NEG),
        Vector3f (POS, NEG, POS),
        Vector3f (POS, POS, NEG),
        Vector3f (POS, NEG, NEG),
        Vector3f (POS, NEG, POS),
        Vector3f (POS, POS, NEG),
        Vector3f (POS, NEG, POS),
        Vector3f (POS, POS, POS),
        Vector3f (NEG, POS, POS),
        Vector3f (NEG, NEG, POS),
        Vector3f (NEG, NEG, NEG),
        Vector3f (NEG, POS, POS),
        Vector3f (NEG, NEG, NEG),
        Vector3f (NEG, POS, NEG),
        Vector3f (NEG, POS, POS),
        Vector3f (NEG, NEG, POS),
        Vector3f (POS, NEG, POS),
        Vector3f (NEG, POS, POS),
        Vector3f (POS, NEG, POS),
        Vector3f (POS, POS, POS),
        Vector3f (POS, POS, NEG),
        Vector3f (POS, NEG, NEG),
        Vector3f (NEG, NEG, NEG),
        Vector3f (POS, POS, NEG),
        Vector3f (NEG, NEG, NEG),
        Vector3f (NEG, POS, NEG) };

#undef POS
#undef NEG

  Skybox::Skybox (const std::string& name, const SkyboxImages& images) :
      m_program ("./vmd-1.8.7/ExscitechResources/Shaders/SkyboxShader.vsh",
          "./vmd-1.8.7/ExscitechResources/Shaders/SkyboxShader.fsh"), m_texUnit (
          0)
  {
    m_texture = Texture::createCubeMap (name, images.images);
    generateBuffers ();
  }

  Skybox::~Skybox ()
  {
    glDeleteBuffers (1, &m_vertexBufferId);
  }

  void
  Skybox::draw (Camera* camera)
  {
    m_program.enable ();
    m_program.setUniform (m_program.getUniformLocation ("g_world"),
        getTransform4x4 ());
    m_program.setUniform (m_program.getUniformLocation ("g_view"),
        camera->getView ());
    m_program.setUniform (m_program.getUniformLocation ("g_projection"),
        camera->getProjection ());

    glActiveTexture (GL_TEXTURE0 + m_texUnit);
    m_program.setUniform (m_program.getUniformLocation ("g_sampler"),
        m_texture->getTextureID ());

    m_program.setAttribPointer (m_vertexBufferId,
        m_program.getAttribLocation ("g_position"), 3, GL_FLOAT, GL_FALSE, 0, 0,
        0);

    glDrawArrays (GL_TRIANGLES, 0, 36);

    m_program.disableAttribute("g_position");
  }

  void
  Skybox::generateBuffers ()
  {
    glGenBuffers (1, &m_vertexBufferId);

    glBindBuffer (GL_ARRAY_BUFFER, m_vertexBufferId);
    glBufferData (GL_ARRAY_BUFFER, sizeof(Vector3f) * 36, ms_vertices,
        GL_STATIC_DRAW);
    glBindBuffer (GL_ARRAY_BUFFER, 0);
  }
}
