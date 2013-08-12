#ifndef SHADERUTILITY_HPP_
#define SHADERUTILITY_HPP_

#include <string>

namespace Exscitech
{

  class ShaderUtility
  {
  public:

    static std::string
    readShaderSource (const std::string& filename);

    static void
    writeShaderInfoLog (GLuint shader, const std::string& logFilename);

  private:

    ShaderUtility ();

    ~ShaderUtility ();

  };
}
#endif
