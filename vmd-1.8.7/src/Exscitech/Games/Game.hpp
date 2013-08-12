#ifndef GAME_HPP_
#define GAME_HPP_

#include <QtCore/QObject>

class VMDApp;

namespace Exscitech
{
  class Game : public QObject
  {
  Q_OBJECT
  public:

    Game ();

    virtual
    ~Game ();

    virtual void
    initWindow () = 0;

    virtual void
    update () = 0;

    virtual void
    handleKeyboardInput (int keyCode) = 0;

    virtual void
    handleKeyboardUp (int key) = 0;

    virtual bool
    handleMouseInput (int screenX, int screenY, int button) = 0;

    virtual bool
    handleMouseMove (int screenX, int screenY)
    {
      return false;
    }

    virtual bool
    handleMouseRelease (int screenX, int screenY, int button) = 0;

    virtual bool
    handleMouseWheel (int delta)
    {
      return false;
    }

    virtual bool
    handleWindowResize (int width, int height)
    {
      return false;
    }

    virtual void
    drawGameGraphics () = 0;

  };
}

#endif
