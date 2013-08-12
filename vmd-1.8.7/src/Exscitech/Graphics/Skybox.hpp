#ifndef SKYBOX_HPP_
#define SKYBOX_HPP_

#include <vector>
#include <string>

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Graphics/Lighting/Texture.hpp"

namespace Exscitech
{
  class Material;

  class Skybox : public Drawable
  {
  public:
    struct SkyboxImages
    {
      enum Direction
      {
        RIGHT, LEFT, BOTTOM, TOP, BACK, FRONT
      };
      enum SkyboxType
      {
        SKY, SPACE
      };
      SkyboxImages (const std::string& right, const std::string& left,
          const std::string& bottom, const std::string& top,
          const std::string& back, const std::string& front)
      {
        images[RIGHT] = right;
        images[LEFT] = left;
        images[BOTTOM] = bottom;
        images[TOP] = top;
        images[BACK] = back;
        images[FRONT] = front;
      }

      SkyboxImages (SkyboxType type)
      {
        std::string base = "./vmd-1.8.7/ExscitechResources/Skybox/";

        switch (type)
        {
          case SKY:
            images[RIGHT] = base + "SkyRight.tga";
            images[LEFT] = base + "SkyLeft.tga";
            images[BOTTOM] = base + "SkyBottom.tga";
            images[TOP] = base + "SkyTop.tga";
            images[BACK] = base + "SkyBack.tga";
            images[FRONT] = base + "SkyFront.tga";
            break;

          case SPACE:
            images[RIGHT] = base + "Galaxy_RT.bmp";
            images[LEFT] = base + "Galaxy_LT.bmp";
            images[BOTTOM] = base + "Galaxy_DN.bmp";
            images[TOP] = base + "Galaxy_UP.bmp";
            images[BACK] = base + "Galaxy_BK.bmp";
            images[FRONT] = base + "Galaxy_FT.bmp";
            break;

        }

      }

      std::string images[6];
    };
  public:

    Skybox (const std::string& name,
        const SkyboxImages& images = SkyboxImages (SkyboxImages::SKY));

    ~Skybox();

    virtual void
    draw (Camera* camera);

    void
    setTexUnit(unsigned int texUnit);

  private:

    static const Vector3f ms_vertices[36];

  private:

    void
    generateBuffers ();

  private:
    uint m_vertexBufferId;
    ShaderProgram m_program;
    Texture* m_texture;
    unsigned int m_texUnit;

  };
}
#endif
