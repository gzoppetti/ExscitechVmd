#ifndef SHADER_PROGRAM_HPP_
#define SHADER_PROGRAM_HPP_

#include <GL/glew.h>
#include <string>
#include <map>

#include "Exscitech/Math/Vector2.hpp"
#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"
#include "Exscitech/Math/Matrix4x4.hpp"

namespace Exscitech
{
  class ShaderProgram
  {
  public:

    ShaderProgram (const std::string& vert, const std::string& frag);

    ShaderProgram (const std::string& vert, const std::string& frag,
        const std::string& geometry);

    virtual
    ~ShaderProgram ();

    void
    writeInfoLog (const std::string& logFilename) const;

    GLint
    getUniformLocation (const char* name);

    void
    setUniform (GLint location, int value);

    void
    setUniform (GLint location, float value);

    void
    setUniform (GLint location, const Vector2f& value);

    void
    setUniform (GLint location, const Vector3f& value);

    void
    setUniform (GLint location, const Vector4f& value);

    void
    setUniform (GLint location, const Matrix4x4f& value);

    virtual void
    enable () const;

    virtual void
    disable () const;

    GLuint
    getId () const;

    GLint
    getNumActiveUniforms () const;

    // Attributes
    GLint
    getNumActiveAttributes () const;

    GLint
    getAttribLocation (const std::string& attributeName) const;

    void
    setAttribPointer (uint bufferId, GLint location, uint size, GLenum type,
        GLboolean normalized, uint stride, uint offset, uint attribDivisor);

    void
    setAttribPointer (GLint location, uint size, GLenum type,
        GLboolean normalized, uint stride, uint attribDivisor, void* data);

    void
    setAttribute (GLint location, float value) const;

    void
    setAttribute (GLint locatione, const Vector2f& value) const;

    void
    setAttribute (GLint location, const Vector3f& value) const;

    void
    setAttribute (GLint location, const Vector4f& value) const;

    void
    setAttribute (GLint location, const Matrix4x4f& value) const;

    void
    enableAttribute(GLint location);

    void
    enableAttribute(const std::string& attrib);

    void
    disableAttribute(GLint location);

    void
    disableAttribute(const std::string& attrib);

    void
    link ();

    GLuint m_programId;
  };
}

#endif
