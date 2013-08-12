#ifndef SCENE_HPP
#define SCENE_HPP

#include <map>
#include <string>

#include "Displayable.h"

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Types.hpp"

namespace Exscitech
{
  class Scene
  {
  public:

    Scene ();
    ~Scene ();

    void
    addDrawable (const std::string& name, Drawable* const drawable);

    void
    addDisplayable (const std::string& name, Displayable* const displayable);

    Drawable*
    removeDrawable(const std::string& name);

    Displayable*
    removeDisplayable(const std::string& name);

    void
    clear();

    void
    render (Camera* camera) const;

  private:

    typedef std::map<std::string, Drawable* const > NameDrawableMap;
    typedef NameDrawableMap::iterator NameDrawableIter;
    typedef NameDrawableMap::const_iterator NameDrawableConstIter;

    typedef std::map<std::string,Displayable* const > NameDisplayableMap;
    typedef NameDisplayableMap::iterator NameDisplayableIter;
    typedef NameDisplayableMap::const_iterator NameDisplayableConstIter;

  private:

    NameDrawableMap m_drawables;
    NameDisplayableMap m_displayables;

  };
}
#endif
