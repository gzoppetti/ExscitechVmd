#include <GL/glew.h>

#include <fstream>

#include "Exscitech/Graphics/Shaders/ShaderUtility.hpp"

#include "Exscitech/Utilities/DebuggingUtility.hpp"

namespace Exscitech
{
  ShaderUtility::ShaderUtility ()
  {
  }

  ShaderUtility::~ShaderUtility ()
  {
  }

  std::string
  ShaderUtility::readShaderSource (const std::string& filename)
  {
    std::ifstream inFile (filename.c_str ());
    std::string fileString ((std::istreambuf_iterator<char> (inFile)),
        std::istreambuf_iterator<char> ());

    return (fileString);
  }

  void
  ShaderUtility::writeShaderInfoLog (GLuint shader,
      const std::string& logFilename)
  {
    GLint infoLogLength = 0;
    glGetShaderiv (shader, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0)
    {
      char* infoLog = new char[infoLogLength];

      glGetShaderInfoLog (shader, infoLogLength, NULL, infoLog);
      std::ofstream logFile (logFilename.c_str ());
      logFile << infoLog << std::endl;
      logFile.close ();

      fprintf(stderr, "%s\n", infoLog);
      delete[] infoLog;
    }
  }
}
