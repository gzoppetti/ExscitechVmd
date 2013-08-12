#ifndef DEBUGGINGUTILITY_HPP_
#define DEBUGGINGUTILITY_HPP_

#include <GL/glew.h>
#include <cstdio>

class DebuggingUtility
{
public:

  static void
  printGlError()
  {
    GLenum error = glGetError();
    const GLubyte* string = gluErrorString(error);
    fprintf(stderr, "Error: %s\n", string);
  }

private:
  DebuggingUtility()
  {
  }

};
#endif
