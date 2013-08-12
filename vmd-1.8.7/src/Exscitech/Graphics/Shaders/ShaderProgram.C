#include <GL/glew.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include <string>

#include "Exscitech/Graphics/Shaders/Shader.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"

namespace Exscitech
{
  ShaderProgram::ShaderProgram (const std::string& vert,
      const std::string& frag)
  {
    m_programId = glCreateProgram ();

    if (m_programId == 0)
    {
      fprintf (stderr, "Error creating Program!\n");
    }

    fprintf (stderr, "Loading Vert Shader: %s\n", vert.c_str ());

    Shader* vertShader = Shader::acquire (vert, Shader::VERTEX_SHADER);
    glAttachShader (m_programId, vertShader->getId ());

    fprintf (stderr, "Loading Frag Shader: %s\n", frag.c_str ());

    Shader* fragShader = Shader::acquire (frag, Shader::FRAGMENT_SHADER);
    glAttachShader (m_programId, fragShader->getId ());

    link ();

    glDetachShader (m_programId, vertShader->getId ());
    glDetachShader (m_programId, fragShader->getId ());
  }

  ShaderProgram::ShaderProgram (const std::string& vert,
      const std::string& frag, const std::string& geometry)
  {
    m_programId = glCreateProgram ();

    if (m_programId == 0)
    {
      fprintf (stderr, "Error creating Program!\n");
    }

    fprintf (stdout, "Loading Vert Shader: %s\n", vert.c_str ());
    Shader* vertShader = Shader::acquire (vert, Shader::VERTEX_SHADER);
    glAttachShader (m_programId, vertShader->getId ());

    fprintf (stdout, "Loading Frag Shader: %s\n", frag.c_str ());
    Shader* fragShader = Shader::acquire (frag, Shader::FRAGMENT_SHADER);
    glAttachShader (m_programId, fragShader->getId ());

    fprintf (stdout, "Loading Geometry Shader: %s\n", geometry.c_str ());
    Shader* geometryShader = Shader::acquire (geometry,
        Shader::GEOMETRY_SHADER);
    glAttachShader (m_programId, geometryShader->getId ());

    link ();

    glDetachShader (m_programId, vertShader->getId ());
    glDetachShader (m_programId, fragShader->getId ());
    glDetachShader (m_programId, geometryShader->getId ());
  }

  ShaderProgram::~ShaderProgram ()
  {
    glDeleteProgram (m_programId);
  }

  GLint
  ShaderProgram::getUniformLocation (const char* name)
  {
    return glGetUniformLocation (m_programId, name);
  }

  void
  ShaderProgram::setUniform (GLint location, int value)
  {
    glUniform1i (location, value);
  }

  void
  ShaderProgram::setUniform (GLint location, float value)
  {
    glUniform1f (location, value);
  }

  void
  ShaderProgram::setUniform (GLint location, const Vector2f& value)
  {
    glUniform2fv (location, 1, &value[0]);
  }

  void
  ShaderProgram::setUniform (GLint location, const Vector3f& value)
  {
    glUniform3fv (location, 1, &value[0]);
  }

  void
  ShaderProgram::setUniform (GLint location, const Vector4f& value)
  {
    glUniform4fv (location, 1, &value[0]);
  }

  void
  ShaderProgram::setUniform (GLint location, const Matrix4x4f& value)
  {
    glUniformMatrix4fv (location, 1, GL_FALSE, &value (0, 0));
  }

  void
  ShaderProgram::link ()
  {
    glLinkProgram (m_programId);

    GLsizei length;
    GLchar info[10000];
    glGetProgramInfoLog (m_programId, 10000, &length, info);
    if (length > 0)
    {
      fprintf (stderr, "Error during link!\n");
      fprintf (stdout, "%s\n", info);
    }
    else
    {
      fprintf (stdout, "Getting uniforms\n");
      GLint numUniforms = getNumActiveUniforms ();
      fprintf (stdout, "Num: %i\n", numUniforms);

      const GLsizei BUFFER_SIZE = 64;
      GLchar nameBuffer[BUFFER_SIZE];
      for (int i = 0; i < numUniforms; ++i)
      {
        GLint size;
        GLenum type;
        glGetActiveUniform (m_programId, i, BUFFER_SIZE, NULL, &size, &type,
            nameBuffer);
        //GLint location = glGetUniformLocation (m_programId, nameBuffer);
        std::string name (nameBuffer);
        fprintf (stdout, "%s\n", name.c_str ());
      }
    }
  }

  void
  ShaderProgram::enable () const
  {
    glUseProgram (m_programId);
  }

  void
  ShaderProgram::disable () const
  {
    glUseProgram (0);
  }

  GLuint
  ShaderProgram::getId () const
  {
    return (m_programId);
  }

  GLint
  ShaderProgram::getNumActiveUniforms () const
  {
    GLint numUniforms;
    glGetProgramiv (m_programId, GL_ACTIVE_UNIFORMS, &numUniforms);
    return (numUniforms);
  }

// Attributes
  GLint
  ShaderProgram::getNumActiveAttributes () const
  {
    GLint numAttributes;
    glGetProgramiv (m_programId, GL_ACTIVE_ATTRIBUTES, &numAttributes);
    return (numAttributes);
  }

  void
  ShaderProgram::setAttribPointer (uint bufferId, GLint location, uint size,
      GLenum type, GLboolean normalized, uint stride, uint offset,
      uint attribDivisor)
  {
    glBindBuffer (GL_ARRAY_BUFFER, bufferId);
    glVertexAttribPointer (location, size, type, normalized, stride,
        (GLvoid*) offset);
    glVertexAttribDivisor (location, attribDivisor);
    glBindBuffer (GL_ARRAY_BUFFER, 0);
  }

  void
  ShaderProgram::setAttribPointer (GLint location, uint size, GLenum type,
      GLboolean normalized, uint stride, uint attribDivisor, void* data)
  {
    glVertexAttribPointer (location, size, type, normalized, stride, data);
    glVertexAttribDivisor (location, attribDivisor);
  }

  GLint
  ShaderProgram::getAttribLocation (const std::string& attributeName) const
  {
    GLint location = glGetAttribLocation (m_programId, attributeName.c_str ());
    glEnableVertexAttribArray (location);
    return (location);
  }

  void
  ShaderProgram::setAttribute (GLint location, float value) const
  {
    glVertexAttrib1f (location, value);
  }

  void
  ShaderProgram::setAttribute (GLint location, const Vector2f& value) const
  {
    glVertexAttrib2fv (location, &value[0]);
  }

  void
  ShaderProgram::setAttribute (GLint location, const Vector3f& value) const
  {
    glVertexAttrib3fv (location, &value[0]);
  }

  void
  ShaderProgram::setAttribute (GLint location, const Vector4f& value) const
  {
    glVertexAttrib4fv (location, &value[0]);
  }

  void
  ShaderProgram::setAttribute (GLint location, const Matrix4x4f& value) const
  {
    glVertexAttrib4fv (location + 0, &value (0, 0));
    glVertexAttrib4fv (location + 1, &value (1, 0));
    glVertexAttrib4fv (location + 2, &value (2, 0));
    glVertexAttrib4fv (location + 3, &value (3, 0));
  }

  void
  ShaderProgram::enableAttribute (GLint location)
  {
    glEnableVertexAttribArray (location);
  }

  void
  ShaderProgram::enableAttribute (const std::string& attrib)
  {
    int location = getAttribLocation (attrib);
    glEnableVertexAttribArray (location);
  }

  void
  ShaderProgram::disableAttribute (GLint location)
  {
    glDisableVertexAttribArray (location);
  }

  void
  ShaderProgram::disableAttribute (const std::string& attrib)
  {
    int location = getAttribLocation (attrib);
    glDisableVertexAttribArray (location);
  }

}
