#include "SSAO.hpp"

namespace Exscitech
{
  const float SSAO::ms_vertices[18] =
    { -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };

  SSAO::SSAO (unsigned int width, unsigned int height) :
      m_fboId (0), m_colorTexId (0), m_depthTexId (0), m_program (
          "./vmd-1.8.7/ExscitechResources/Shaders/SSAOShader.vsh",
          "./vmd-1.8.7/ExscitechResources/Shaders/SSAOShader.fsh")
  {
    m_positionAttributeLocation = m_program.getAttribLocation ("g_position");
    m_screenSizeUniformLocation = m_program.getUniformLocation ("screenSize");
    m_cameraRangeUniformLocation = m_program.getUniformLocation ("cameraRange");
    m_depthTextureLocation = m_program.getUniformLocation ("depthTexture");
    m_randomTextureLocation = m_program.getUniformLocation ("randomTexture");
//    m_sceneTextureLocation = m_program.getUniformLocation("sceneTexture");

    glGenBuffers (1, &m_vboId);
    glBindBuffer (GL_ARRAY_BUFFER, m_vboId);
    glBufferData (GL_ARRAY_BUFFER, sizeof(float) * 18, ms_vertices,
        GL_STATIC_DRAW);

    glBindBuffer (GL_ARRAY_BUFFER, 0);

    glGenFramebuffers (1, &m_fboId);
    resize (width, height);

    enable ();
    checkFboStatus ();
    disable ();

    generateRandomTexture ();
  }

  SSAO::~SSAO ()
  {
    glDeleteBuffers(1, &m_vboId);
    glDeleteTextures (1, &m_colorTexId);
    glDeleteTextures (1, &m_depthTexId);
    glDeleteTextures (1, &m_randomTexId);
    glDeleteFramebuffers (1, &m_fboId);
  }

  void
  SSAO::generateRandomTexture ()
  {
    glGenTextures (1, &m_randomTexId);
    glActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, m_randomTexId);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    Vector3f random[128 * 128];

    for (unsigned int row; row < 128; ++row)
    {
      for (unsigned int col; col < 128; ++col)
      {
        // u [0..1)
        int u = rand () % 100 / 100.f;

        // v [-1..1)
        int v = rand () % 100 / 50.f - 1.0f;
        float theta = 2 * M_PI * u;
        float phi = std::acos (v);
        random[col + row * 128].set (sin (theta) * cos (phi),
            sin (theta) * sin (phi), cos (theta));
        random[col + row * 128].normalize();
      }
    }

    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB8, 128, 128, 0, GL_RGB,
        GL_FLOAT, random);

    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void
  SSAO::checkFboStatus ()
  {
    fprintf (stderr, "Scene Texture: D %u F %u C %u\n", m_depthTexId, m_fboId,
        m_colorTexId);

    // check FBO status
    GLenum status = glCheckFramebufferStatus (GL_FRAMEBUFFER);
    switch (status)
    {
      case GL_FRAMEBUFFER_COMPLETE:
        fprintf (stderr, "Framebuffer complete.\n");
        break;

      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        fprintf (stderr,
            "[ERROR] Framebuffer incomplete: Attachment is NOT complete.");
        break;

      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        fprintf (stderr,
            "[ERROR] Framebuffer incomplete: No image is attached to FBO.");
        break;

      case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
        fprintf (stderr,
            "[ERROR] Framebuffer incomplete: Attached images have different dimensions.");
        break;

      case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        fprintf (stderr,
            "[ERROR] Framebuffer incomplete: Color attached images have different internal formats.");
        break;

      default:
        fprintf (stderr, "[ERROR] Unknown error.");
        break;
    }
    if (status != GL_FRAMEBUFFER_COMPLETE)
      fprintf (stderr, "Frame buffer is not complete!\n");
  }

  void
  SSAO::resize (unsigned int width, unsigned int height)
  {
    m_width = width;
    m_height = height;

    m_program.enable ();
    m_program.setUniform (m_screenSizeUniformLocation,
        Vector2f (width, height));
    m_program.disable ();

    if (m_colorTexId != 0)
      glDeleteTextures (1, &m_colorTexId);
    if (m_depthTexId != 0)
      glDeleteTextures (1, &m_depthTexId);

    glActiveTexture (GL_TEXTURE0);
    glGenTextures (1, &m_colorTexId);
    glGenTextures (1, &m_depthTexId);

    glBindTexture (GL_TEXTURE_2D, m_colorTexId);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, NULL);

    glBindTexture (GL_TEXTURE_2D, m_depthTexId);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, m_width, m_height, 0,
        GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    enable ();
    glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
        m_colorTexId, 0);
    glFramebufferTexture2D (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
        m_depthTexId, 0);
    disable ();
  }

  void
  SSAO::enable ()
  {
    glGetIntegerv (GL_FRAMEBUFFER_BINDING, &m_oldFboBinding);
    glBindFramebuffer (GL_FRAMEBUFFER, m_fboId);
    glClearColor (0, 0, 0, 0);
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }

  void
  SSAO::disable ()
  {
    glBindFramebuffer (GL_FRAMEBUFFER, m_oldFboBinding);
  }

  unsigned int
  SSAO::getColorTexture ()
  {
    return m_colorTexId;
  }

  unsigned int
  SSAO::getDepthTexture ()
  {
    return m_depthTexId;
  }
  void
  SSAO::draw (Camera* camera)
  {
    glEnable (GL_TEXTURE_2D);
    m_program.enable ();
    m_program.setUniform (m_cameraRangeUniformLocation, camera->getNearFar ());

    m_program.enableAttribute (m_positionAttributeLocation);
    m_program.setAttribPointer (m_vboId, m_positionAttributeLocation, 3,
        GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, 0);

    glActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, m_depthTexId);
    glActiveTexture (GL_TEXTURE1);
    glBindTexture (GL_TEXTURE_2D, m_colorTexId);

    m_program.setUniform (m_depthTextureLocation, 0);
    m_program.setUniform (m_randomTextureLocation, 1);
    glDrawArrays (GL_TRIANGLES, 0, 6);

    glActiveTexture (GL_TEXTURE0);
    glBindTexture (GL_TEXTURE_2D, 0);
    glActiveTexture (GL_TEXTURE1);
    glBindTexture (GL_TEXTURE_2D, 0);

    m_program.disableAttribute (m_positionAttributeLocation);
    m_program.disable ();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_DST_COLOR, GL_ZERO);
    m_quad.setTexture(m_colorTexId);
    m_quad.draw(0);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
  }
}
