#ifndef SHADER_HPP_
#define SHADER_HPP_
#include <GL/glew.h>

#include <string>
#include <map>
#include <stdio.h>
#include <boost/filesystem.hpp>

#include "Exscitech/Graphics/Shaders/ShaderUtility.hpp"

namespace bf = boost::filesystem;

namespace Exscitech
{
  class Shader
  {

  public:

    enum ShaderType
    {
      VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER
    };

  private:

    typedef std::map<std::string, Shader*> ShaderMapType;

  public:

    static Shader*
    acquire (const std::string& srcFilename,
        ShaderType shaderType);

    static void
    release (const std::string& srcFilename);

    static void
    clearMap ();

  public:

    GLuint
    getId () const;

    ShaderType
    getType () const;

  private:

    Shader (ShaderType shaderType, const std::string& filename);

    ~Shader ();

    void
    compile (const std::string& srcFilename) const;

    void
    writeInfoLog (const std::string& srcFilename) const;

  private:

    static ShaderMapType ms_shaderMap;

  private:

    GLuint m_shaderId;
    ShaderType m_shaderType;

  };
}
#endif
