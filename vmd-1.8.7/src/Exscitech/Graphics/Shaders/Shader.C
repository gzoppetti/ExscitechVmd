#include "Exscitech/Graphics/Shaders/Shader.hpp"

namespace Exscitech
{
  Shader::ShaderMapType Shader::ms_shaderMap;

  Shader*
  Shader::acquire (const std::string& srcFilename, ShaderType shaderType)
  {
    ShaderMapType::const_iterator iter = ms_shaderMap.find (srcFilename);
    if (iter != ms_shaderMap.end ())
    {
      return (iter->second);
    }
    Shader* shader = new Shader (shaderType, srcFilename);
    ms_shaderMap[srcFilename] = shader;

    return (shader);
  }

  void
  Shader::release (const std::string& srcFilename)
  {
    ShaderMapType::iterator iter = ms_shaderMap.find (srcFilename);
    if (iter != ms_shaderMap.end ())
    {
      delete iter->second;
      ms_shaderMap.erase (iter);
    }
  }

  void
  Shader::clearMap ()
  {
    for (ShaderMapType::iterator iter = ms_shaderMap.begin();
        iter != ms_shaderMap.end (); ++iter)
    {
      delete iter->second;
    }
    ms_shaderMap.clear();

  }
  GLuint
  Shader::getId () const
  {
    return (m_shaderId);
  }

  Shader::ShaderType
  Shader::getType () const
  {
    return (m_shaderType);
  }

  Shader::Shader (ShaderType shaderType, const std::string& filename) :
      m_shaderType (shaderType)
  {
    GLuint glShaderType;
    switch (m_shaderType)
    {
      case VERTEX_SHADER:
        glShaderType = GL_VERTEX_SHADER;
        break;
      case GEOMETRY_SHADER:
        glShaderType = GL_GEOMETRY_SHADER;
        break;
      case FRAGMENT_SHADER:
        glShaderType = GL_FRAGMENT_SHADER;
        break;
    }
    m_shaderId = glCreateShader (glShaderType);
    compile (filename);
    writeInfoLog (filename);
  }

  Shader::~Shader ()
  {
    glDeleteShader (m_shaderId);
  }

  void
  Shader::compile (const std::string& srcFilename) const
  {
    std::string shaderSource = ShaderUtility::readShaderSource (srcFilename);
    const char* shaderSrcPtr = shaderSource.c_str ();

    glShaderSource (m_shaderId, 1, &shaderSrcPtr, NULL);
    glCompileShader (m_shaderId);
  }

  void
  Shader::writeInfoLog (const std::string& srcFilename) const
  {
    //std::string baseName = bf::basename (srcFilename);
    std::string infoLogFilename (srcFilename + ".log");
    ShaderUtility::writeShaderInfoLog (m_shaderId, infoLogFilename);
  }
}
