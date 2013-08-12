#ifndef QT_OPENGL_DISPLAY_DEVICE_HPP_
#define QT_OPENGL_DISPLAY_DEVICE_HPP_

#include <QtGui/QWidget>
#include <QtOpenGL/QGLWidget>

#include "OpenGLRenderer.h"

namespace Exscitech
{
  class VmdGlWidget;

  class QtOpenGLDisplayDevice : public OpenGLRenderer
  {
  public:

    QtOpenGLDisplayDevice (VMDApp* vmdApp, int* windowSize = NULL,
        int* windowOrigin = NULL);

    virtual
    ~QtOpenGLDisplayDevice ();

    VmdGlWidget*
    getGlWindow ();

    //
    // get the current state of the device's pointer (i.e. cursor if it has one)
    //

    // Last Qt event we processed
    int m_lastEvent;
    // Last mouse button pressed
    int m_lastButton;
    // Last mouse wheel delta (Windows parlance)
    int m_lastZDelta;

    int m_keyboardModifiers;

    int m_lastMouseX;
    int m_lastMouseY;

    virtual void
    makeContextCurrent ();

    // abs pos of cursor from lower-left corner
    virtual int
    x ();

    // same, for y direction
    virtual int
    y ();

    virtual int
    shift_state ();

    virtual int
    spaceball (int* rx, int* ry, int* rz, int* tx, int* ty, int* tz,
        int* buttons);

    virtual void
    set_cursor (int n);

    virtual int
    read_event (long& retdev, long& retval);

    virtual void
    reshape ();

    virtual unsigned char*
    readpixels (int& x, int& y);

    virtual void
    update (int do_update = TRUE);

    virtual void
    do_resize_window (int w, int h);

    // Update xOrig and yOrig before computing screen position
    virtual void
    rel_screen_pos (float& x, float& y)
    {
      reshape ();
      DisplayDevice::rel_screen_pos (x, y);
    }

  private:

    void
    doVmdInitialization ();

  private:

    VmdGlWidget* m_vmdWidget;

  };

}

#endif
