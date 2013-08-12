#ifndef SSAO_HPP_
#define SSAO_HPP_

#include <GL/glew.h>

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Graphics/FullQuad.hpp"

namespace Exscitech
{
  class SSAO : public Drawable
  {
  public:

    SSAO (unsigned int width, unsigned int height);

    ~SSAO ();

    void
    resize (unsigned int width, unsigned int height);

    void
    enable ();

    void
    disable ();

    unsigned int
    getColorTexture();

    unsigned int
    getDepthTexture();

    void
    draw (Camera* camera);

  private:

    void
    checkFboStatus ();

    void
    generateRandomTexture();

  private:
    static const float ms_vertices[18];

  private:

    unsigned int m_width;
    unsigned int m_height;

    unsigned int m_fboId;
    unsigned int m_colorTexId;
    unsigned int m_depthTexId;
    unsigned int m_vboId;

    unsigned int m_randomTexId;

    int m_oldFboBinding;

    FullQuad m_quad;
    ShaderProgram m_program;
    int m_positionAttributeLocation;
    int m_cameraRangeUniformLocation;
    int m_screenSizeUniformLocation;
    int m_depthTextureLocation;
    int m_randomTextureLocation;
  };
}

#endif
