#ifndef IDRAWABLE_HPP_
#define IDRAWABLE_HPP_

#include "Exscitech/Display/Camera.hpp"

namespace Exscitech
{
  class IDrawable
  {
  public:

    virtual
    ~IDrawable ()
    {
    }

    virtual void
    draw (Camera* camera) = 0;
  };
}

#endif
