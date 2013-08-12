#include <string>
#include <vector>

#include <GL/glew.h>

#include "Texture.hpp"
#include "CubeMap.hpp"
#include "Exscitech/Types.hpp"
#include "Exscitech/Utilities/DebuggingUtility.hpp"

namespace Exscitech
{
  Texture::TextureMap Texture::ms_textureMap;

  //**

  Texture*
  Texture::create (const std::string& name, const std::string& fileName)
  {
    Texture::TextureIterator i = ms_textureMap.find (fileName);
    if (i != ms_textureMap.end ())
    {
      return (i->second);
    }

    Texture* texture = new Texture (name, fileName);
    ms_textureMap.insert (make_pair (fileName, texture));
    return (texture);
  }

  Texture*
  Texture::createCubeMap (const std::string& name, const std::string* fileNames)
  {
    Texture::TextureIterator i = ms_textureMap.find (name);
    if (i != ms_textureMap.end ())
    {
      return (i->second);
    }

    Texture* texture = new CubeMap (name, fileNames);

    ms_textureMap.insert (make_pair (name, texture));
    return (texture);
  }

  void
  Texture::release (const std::string& name)
  {
    TextureMap::iterator iter;
    iter = ms_textureMap.find (name);
    if (iter != ms_textureMap.end ())
    {
      // Texture was found
      Texture* texture = iter->second;
      delete texture;
      ms_textureMap.erase (iter);
    }
  }

  void
  Texture::clearMap ()
  {
    for (TextureMap::iterator iter = ms_textureMap.begin ();
        iter != ms_textureMap.end (); ++iter)
    {
      delete iter->second;
    }
    ms_textureMap.clear();

  }
  //**

  void
  Texture::setSize (GLuint width, GLuint height)
  {
    m_width = width;
    m_height = height;
  }

  void
  Texture::setDepth (GLuint bytesPerPixel)
  {
    m_bytesPerPixel = bytesPerPixel;
  }

  void
  Texture::enable (uchar textureUnit) const
  {
    glActiveTexture (GL_TEXTURE0 + textureUnit);
    glBindTexture (GL_TEXTURE_2D, m_textureId);
  }

  void
  Texture::disable () const
  {
    glBindTexture (GL_TEXTURE_2D, 0);
    glDisable (GL_TEXTURE_2D);
  }

  void
  Texture::setTextureID (GLint id)
  {
    m_textureId = id;
  }

  GLint
  Texture::getTextureID () const
  {
    return (m_textureId);
  }

  Texture::Texture (const std::string& name, GLuint width, GLuint height) :
      m_name (name)
  {
    m_width = width;
    m_height = height;
  }

  Texture::Texture (const std::string& name, const std::string& fileName) :
      m_name (name)
  {

    ILuint image = loadImage (fileName);
    generateTexture (image);
  }

  Texture::Texture ()
  {
  }

  Texture::~Texture ()
  {
    glDeleteTextures (1, &m_textureId);
  }

  ILuint
  Texture::loadImage (const std::string& fileName)
  {
    //m_texID = ilutGLLoadImage ((char *)m_fileName.c_str ());

    ILuint image;
    ilGenImages (1, &image);
    ilBindImage (image);
    bool loadedImage = ilLoadImage (fileName.c_str ());

    if (!loadedImage)
    {
      fprintf (stderr, "Devil failed to load the image! %s\n",
          fileName.c_str ());
    }

    m_width = ilGetInteger (IL_IMAGE_WIDTH);
    m_height = ilGetInteger (IL_IMAGE_HEIGHT);
    m_bytesPerPixel = ilGetInteger (IL_IMAGE_BPP);
    // DevIL's type and format match OGL's (so far)
    m_format = ilGetInteger (IL_IMAGE_FORMAT);
    m_type = ilGetInteger (IL_IMAGE_TYPE);

    return (image);
  }

  void
  Texture::generateTexture (ILuint image)
  {
    GLubyte* pixels = ilGetData ();
    glGenTextures (1, &m_textureId);
    glBindTexture (GL_TEXTURE_2D, m_textureId);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST );
    glTexImage2D (GL_TEXTURE_2D, 0, 3, m_width, m_height, 0, m_format, m_type,
        pixels);

    ilDeleteImages (1, &image);
  }

  void
  Texture::setTextureParameters (GLenum target, GLenum name, GLint value)
  {
    glBindTexture (GL_TEXTURE_2D, m_textureId);
    glTexParameteri (target, name, value);
    glBindTexture (GL_TEXTURE_2D, 0);
  }
}
