#ifndef VMD_GL_WIDGET_HPP_
#define VMD_GL_WIDGET_HPP_

#include <QtOpenGL/QGLWidget>

namespace Exscitech
{
  class QtOpenGLDisplayDevice;

  class VmdGlWidget : public QGLWidget
  {

  public:

    VmdGlWidget (QtOpenGLDisplayDevice* displayDevice, QWidget* parent = NULL);

    void
    setVmdRespondKeys(bool respondKeys);

    void
    setVmdRespondMouse(bool respondMouse);

    void
    setVmdRespondWheel(bool respondWheel);

    void
    restoreDefaults();

  protected:

    void
    initializeGL();

    void
    resizeGL(int w, int h);

    void
    paintGL();

    void
    keyPressEvent(QKeyEvent* event);

    void
    mousePressEvent (QMouseEvent* event);

    void
    mouseReleaseEvent (QMouseEvent* event);

    void
    mouseMoveEvent (QMouseEvent* event);

    void
    wheelEvent (QWheelEvent* event);

  private:

    bool m_vmdRespondToKeys;
    bool m_vmdRespondToMouse;
    bool m_vmdRespondToWheel;

    QtOpenGLDisplayDevice* m_displayDevice;


  };

}

#endif
