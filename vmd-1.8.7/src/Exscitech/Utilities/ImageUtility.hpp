#ifndef IMAGEUTILITY_HPP_
#define IMAGEUTILITY_HPP_

#include <GL/glew.h>

#include <IL/il.h>
#include <IL/ilu.h>

namespace Exscitech
{
  namespace ImageUtility
  {
    void saveFrameAsImage(const std::string& imageFile)
    {
      int viewport[4];
      glGetIntegerv(GL_VIEWPORT, viewport);
      int width = viewport[2] - viewport[0];
      int height = viewport[3] - viewport[1];

      unsigned char pixels[width * height * 3];
      glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGB, GL_UNSIGNED_BYTE, pixels);
      ILuint imageId;
      ilGenImages(1, &imageId);
      ilBindImage(imageId);
      ilTexImage(width, height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, pixels);
      ilEnable(IL_FILE_OVERWRITE);
      ilSaveImage(imageFile.c_str());
      ilDeleteImage(imageId);
      ilDisable(IL_FILE_OVERWRITE);
    }
  }
}


#endif
