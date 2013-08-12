#ifndef QT_VMDGLWINDOW_HPP_
#define QT_VMDGLWINDOW_HPP_

#include <QtOpenGL/QGLWidget>

#include "Exscitech/Display/QtOpenGLDisplayDevice.hpp"

class VMDApp;

namespace Exscitech
{
  class QtVmdGlWindow : public QGLWidget
  {
  public:

    QtVmdGlWindow (QtOpenGLDisplayDevice* displayDevice, VMDApp* vmdApp,
        QWidget* parent = NULL);

    virtual
    ~QtVmdGlWindow ();

    void
    setRespondToEvents (bool respondToEvents);

  private:

    void
    initializeGl ();

    void
    paintGl ();

    void
    resizeGl (int width, int height);

    void
    keyPressEvent (QKeyEvent* event);

    void
    keyReleaseEvent (QKeyEvent* event);

    void
    mousePressEvent (QMouseEvent* event);

    void
    mouseReleaseEvent (QMouseEvent* event);

    void
    mouseMoveEvent (QMouseEvent* event);

    void
    wheelEvent (QWheelEvent* event);

  private:

    QtOpenGLDisplayDevice* m_displayDevice;
    VMDApp* m_vmdApp;
    bool m_respondToEvents;

  };

}

#endif 
