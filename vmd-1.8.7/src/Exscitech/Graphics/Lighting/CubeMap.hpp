#ifndef CUBEMAP_HPP_
#define CUBEMAP_HPP_

#include <vector>
#include <string>

#include "Texture.hpp"

namespace Exscitech
{
  class CubeMap : public Texture
  {
  public:

    CubeMap (const std::string& name, const std::string* textureFIles);

    virtual void
    enable (int textureUnit);

    virtual void
    disable ();

  private:

    void
    loadTexturesFromFiles (const std::string* fileNames);

  };
}

#endif
