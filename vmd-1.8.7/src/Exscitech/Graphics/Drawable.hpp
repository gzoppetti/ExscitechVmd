#ifndef DRAWABLE_HPP_
#define DRAWABLE_HPP_
#include <GL/glew.h>
#include <string>

#include "Exscitech/Graphics/IDrawable.hpp"
#include "Exscitech/Graphics/Transformable.hpp"
#include "Exscitech/Display/Camera.hpp"

namespace Exscitech
{
  class Drawable : public IDrawable, public Transformable
  {
  public:

    Drawable ();

    virtual
    ~Drawable ();

    virtual void
    draw (Camera* camera) = 0;
  };
}
#endif
