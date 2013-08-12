#include <GL/glew.h>
//#include <QtOpenGL/QtOpenGL>
#include <QtGui/qevent.h>
#include "config.h"
#include "Exscitech/Display/QtOpenGLDisplayDevice.hpp"
#include "Exscitech/Display/VmdGlWidget.hpp"

#include <cstdio>

namespace Exscitech
{
  // static data for this object
  static const char *glStereoNameStr[OPENGL_STEREO_MODES] =
    {
        "Off",
        "CrystalEyes",
        "CrystalEyesReversed",
        "DTI SideBySide",
        "Scanline Interleaved",
        "Anaglyph",
        "CrossEyes",
        "SideBySide",
        "AboveBelow",
        "Left",
        "Right" };

  static const char *glCacheNameStr[OPENGL_CACHE_MODES] =
    { "Off", "On" };

  //**

  QtOpenGLDisplayDevice::QtOpenGLDisplayDevice (VMDApp* vmdApp,
      int* windowSize, int* windowOrigin) :
    OpenGLRenderer ("VMD " VMDVERSION "OpenGL Display")
  {
    // set up data possible before opening window
    stereoNames = glStereoNameStr;
    stereoModes = OPENGL_STEREO_MODES;

    // GLSL is only available on MacOS X 10.4 so far.
#if defined(__APPLE__)
    renderNames = glRenderNameStr;
    renderModes = OPENGL_RENDER_MODES;
#endif

    cacheNames = glCacheNameStr;
    cacheModes = OPENGL_CACHE_MODES;

    m_vmdWidget = new VmdGlWidget(this);
    m_vmdWidget->show();

    // Exscitech: Context is set up so init GLEW before the setup of OpenGL state
    GLenum glewInitCode = glewInit ();
    if (glewInitCode != GLEW_OK)
    {
      fprintf (stderr, "glewInit Error: %s\n",
          glewGetErrorString (glewInitCode));
      exit (-1);
    }

    doVmdInitialization ();

    m_vmdWidget->hide();
  }

  void
  QtOpenGLDisplayDevice::doVmdInitialization ()
  {
    // stereo is off initially
    ext->hasstereo = FALSE;
    // stereo not forced initially
    ext->stereodrawforced = FALSE;
    // multisample is off initially
    ext->hasmultisample = FALSE;

    setup_initial_opengl_state ();

    // set flags for the capabilities of this display
    // whether we can do antialiasing or not.
    if (ext->hasmultisample)
      // we use multisampling over other methods
      aaAvailable = TRUE;
    else
      // no non-multisample implementation yet
      aaAvailable = FALSE;

    cueingAvailable = TRUE;
    cullingAvailable = TRUE;
    cullingEnabled = FALSE;

    // set default settings
    if (ext->hasmultisample)
    {
      // enable fast multisample based antialiasing by default
      aa_on ();
      // other antialiasing techniques are slow, so only multisample
      // makes sense to enable by default.
    }
    // leave depth cueing off by default, since its a speed hit.
    cueing_off ();

    set_sphere_mode (sphereMode);
    set_sphere_res (sphereRes);
    set_line_width (lineWidth);
    set_line_style (lineStyle);

    screenX = m_vmdWidget->width ();
    screenY = m_vmdWidget->height ();

    // reshape and clear the display, which initializes some other variables
    reshape ();
    normal ();
    clear ();
    update ();
  }

  QtOpenGLDisplayDevice::~QtOpenGLDisplayDevice ()
  {
    free_opengl_ctx ();
    delete m_vmdWidget;
  }

  VmdGlWidget*
  QtOpenGLDisplayDevice::getGlWindow ()
  {
    return m_vmdWidget;
  }

  void
  QtOpenGLDisplayDevice::makeContextCurrent ()
  {
    m_vmdWidget->makeCurrent ();
  }

  //
  // get the current state of the device's pointer (i.e. cursor if it has one)
  //

  // abs pos of cursor from lower-left corner of display
  int
  QtOpenGLDisplayDevice::x ()
  {
    return xOrig + m_lastMouseX;
  }

  // same, for y direction
  int
  QtOpenGLDisplayDevice::y ()
  {
    return screenY - m_lastMouseY + yOrig;
  }

  // return the current state of the shift, control, and alt keys
  int
  QtOpenGLDisplayDevice::shift_state ()
  {
    int retval = 0;
    if (m_keyboardModifiers & Qt::SHIFT)
      retval |= SHIFT;
    if (m_keyboardModifiers & Qt::CTRL)
      retval |= CONTROL;
    if (m_keyboardModifiers & Qt::ALT)
      retval |= ALT;

    return retval;
  }

  // return the spaceball state, if any
  int
  QtOpenGLDisplayDevice::spaceball (int* rx, int* ry, int* rz, int* tx,
      int* ty, int* tz, int* buttons)
  {
    // not implemented yet
    return 0;
  }

  // set the Nth cursor shape as the current one.  If no arg given, the
  // default shape (n=0) is used.
  void
  QtOpenGLDisplayDevice::set_cursor (int n)
  {
    switch (n)
    {
      default:
      case DisplayDevice::NORMAL_CURSOR:
        m_vmdWidget->setCursor (Qt::ArrowCursor);
        break;
      case DisplayDevice::TRANS_CURSOR:
        m_vmdWidget->setCursor (Qt::SizeAllCursor);
        break;
      case DisplayDevice::SCALE_CURSOR:
        m_vmdWidget->setCursor (Qt::SizeHorCursor);
        break;
      case DisplayDevice::PICK_CURSOR:
        m_vmdWidget->setCursor (Qt::CrossCursor);
        break;
      case DisplayDevice::WAIT_CURSOR:
        m_vmdWidget->setCursor (Qt::WaitCursor);
        break;
    }
  }

  //
  // event handling routines
  //

  // read the next event ... returns an event type (one of the above ones),
  // and a value.  Returns success, and sets arguments.
  int
  QtOpenGLDisplayDevice::read_event (long& retdev, long& retval)
  {
    switch (m_lastEvent)
    {
      case QEvent::Wheel:
        // XXX tests on the Mac show that FLTK is using a coordinate system
        // backwards from what is used on Windows' zDelta value.
        if (m_lastZDelta < 0)
        {
          retdev = WIN_WHEELUP;
        }
        else
        {
          retdev = WIN_WHEELDOWN;
        }
        break;
      case QEvent::MouseButtonPress:
      case QEvent::DragMove:
      case QEvent::MouseButtonRelease:
        if (m_lastButton == Qt::LeftButton)
          retdev = WIN_LEFT;
        else if (m_lastButton == Qt::MiddleButton)
          retdev = WIN_MIDDLE;
        else if (m_lastButton == Qt::RightButton)
          retdev = WIN_RIGHT;
        else
        {
          //printf("unknown button: %d\n", lastbtn);
        }
        retval = (m_lastEvent == QEvent::MouseButtonPress || m_lastEvent
            == QEvent::DragMove);
        break;

      case QEvent::KeyPress:
        retdev = WIN_KEYBD;
        retval = m_lastButton;
        break;

      default:
        return 0;
    }
    m_lastEvent = 0;
    return 1;
  }

  //
  // virtual routines for preparing to draw, drawing, and finishing drawing
  //

  // reshape the display after a shape change
  void
  QtOpenGLDisplayDevice::reshape ()
  {
    xSize = m_vmdWidget->width ();
    ySize = m_vmdWidget->height ();
    xOrig = m_vmdWidget->x ();
    yOrig = m_vmdWidget->y ();

    switch (inStereo)
    {
      case OPENGL_STEREO_SIDE:
      case OPENGL_STEREO_CROSSED:
        set_screen_pos (0.5f * (float) xSize / (float) ySize);
        break;

      case OPENGL_STEREO_ABOVEBELOW:
        set_screen_pos (2.0f * (float) xSize / (float) ySize);
        break;

      case OPENGL_STEREO_STENCIL:
        enable_stencil_stereo ();
        set_screen_pos ((float) xSize / (float) ySize);
        break;

      default:
        set_screen_pos ((float) xSize / (float) ySize);
        break;
    }
  }

  unsigned char*
  QtOpenGLDisplayDevice::readpixels (int& x, int& y)
  {
    unsigned char * img;

    x = xSize;
    y = ySize;

    if ((img = (unsigned char *) malloc (x * y * 3)) != NULL)
    {
#if !defined(WIREGL)
      glPixelStorei (GL_PACK_ALIGNMENT, 1);
      glReadPixels (0, 0, x, y, GL_RGB, GL_UNSIGNED_BYTE, img);
#endif
    }
    else
    {
      x = 0;
      y = 0;
    }

    return img;
  }

  // update after drawing
  void
  QtOpenGLDisplayDevice::update (int doUpdate)
  {
    if (doUpdate)
    {
      m_vmdWidget->swapBuffers();
    }

    glDrawBuffer (GL_BACK);
  }

  void
  QtOpenGLDisplayDevice::do_resize_window (int width, int height)
  {
    DisplayDevice::do_resize_window(width, height);
  }

}
