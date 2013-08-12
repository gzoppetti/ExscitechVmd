#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>
#include <map>
#include <vector>

#include <GL/glew.h>
#include <IL/il.h>

#include "Exscitech/Types.hpp"

namespace Exscitech
{
  class Texture
  {
  public:

    static Texture*
    create (const std::string& name, const std::string& fileName);

    static Texture*
    createCubeMap (const std::string& name, const std::string* fileNames);

    static void
    release (const std::string& name);

    static void
    clearMap ();

  public:

    virtual
    ~Texture ();

    void
    setSize (GLuint width, GLuint height);

    void
    setDepth (GLuint bytesPerPixel);

    virtual void
    enable (uchar textureUnit) const;

    virtual void
    disable () const;

    void
    setTextureID (GLint id);

    GLint
    getTextureID () const;

    void
    setTextureParameters (GLenum target, GLenum name, GLint value);

    Texture ();

  protected:

    Texture (const std::string& name, GLuint width, GLuint height);

    ILuint
    loadImage (const std::string& fileName);

  private:

    Texture (const std::string& name, const std::string& fileName);

    void
    generateTexture (ILuint image);

  private:

    typedef std::map<std::string, Texture*> TextureMap;
    typedef TextureMap::iterator TextureIterator;

    static TextureMap ms_textureMap;

  public:

  protected:

    std::string m_name;
    GLuint m_bytesPerPixel;
    GLuint m_width;
    GLuint m_height;
    GLuint m_textureId;
    // GL_UNSIGNED_BYTE, etc.
    GLuint m_type;
    // GL_RGB etc.
    GLuint m_format;
  };
}

#endif
