#include <GL/glew.h>
//#include <IL/ilut.h>

#include "Exscitech/Graphics/Lighting/CubeMap.hpp"

namespace Exscitech
{

  /*
   * Expects 6 valid file names
   */
  CubeMap::CubeMap (const std::string& name, const std::string* textureFiles) :
      Texture ()
  {
    m_name = name;
    loadTexturesFromFiles (textureFiles);
  }

  void
  CubeMap::loadTexturesFromFiles (const std::string* fileNames)
  {
    const uint numberOfFiles = 6;

    ILuint images[numberOfFiles];
    ilGenImages (numberOfFiles, images);

    for (uchar i = 0; i < numberOfFiles; i++)
    {
      ilBindImage (images[i]);
      ILboolean ok = ilLoadImage (fileNames[i].c_str ());
      if (!ok)
      {
        fprintf (stderr, "Failed to load image: %s\n", fileNames[i].c_str ());
      }
    }

    glGenTextures (1, &m_textureId);
    glBindTexture (GL_TEXTURE_CUBE_MAP, m_textureId);

    // set wrapping and filtering
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    ilBindImage (images[0]);
    // width and height must be same for 6 images
    m_width = ilGetInteger (IL_IMAGE_WIDTH);
    m_height = ilGetInteger (IL_IMAGE_HEIGHT);
    m_format = ilGetInteger (IL_IMAGE_FORMAT);

    // handle remaining 5 faces

    for (uchar face = 0; face < numberOfFiles; face++)
    {
      ilBindImage (images[face]);
      ILubyte *pixels = ilGetData ();
      glTexImage2D (GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, 3, m_width,
          m_height, 0, m_format, GL_UNSIGNED_BYTE, pixels);
    }

    ilDeleteImages (numberOfFiles, images);
    //glBindTexture (GL_TEXTURE_CUBE_MAP, 0);
  }

  /*
   * Enables the CubeMap for use in the pipeline.
   */
  void
  CubeMap::enable (int textureUnit)
  {
    glActiveTexture (GL_TEXTURE0 + textureUnit);
    glBindTexture (GL_TEXTURE_CUBE_MAP, m_textureId);
  }

  /*
   * Disables the texture at the currently active texture unit.
   */
  void
  CubeMap::disable ()
  {
    glBindTexture (GL_TEXTURE_CUBE_MAP, 0);
  }

}
