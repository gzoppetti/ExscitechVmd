#include <GL/glew.h>
#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>

#include "VMDApp.h"

#include "Exscitech/Display/QtVmdGlWindow.hpp"
#include "Exscitech/Games/GameController.hpp"

namespace Exscitech
{
  QtVmdGlWindow::QtVmdGlWindow (QtOpenGLDisplayDevice* displayDevice,
      VMDApp* vmdApp, QWidget* parent) :
        QGLWidget (
            QGLFormat (QGL::DepthBuffer | QGL::DoubleBuffer | QGL::Rgba),
            parent), m_displayDevice (displayDevice), m_vmdApp (vmdApp),
        m_respondToEvents (true)
  {
  }

  QtVmdGlWindow::~QtVmdGlWindow ()
  {
    fprintf(stderr, "Constructor Initializing Gl Window\n");
    glFrontFace(GL_CCW);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
  }

  void
  QtVmdGlWindow::setRespondToEvents (bool respondToEvents)
  {
    m_respondToEvents = respondToEvents;
  }

  void
  QtVmdGlWindow::initializeGl ()
  {
    fprintf(stderr, "initializing gl window\n");
    glFrontFace(GL_CCW);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
  }

  void
  QtVmdGlWindow::paintGl ()
  {
    m_displayDevice->reshape ();
    m_displayDevice->_needRedraw = 1;

    m_vmdApp->VMDupdate (VMD_IGNORE_EVENTS);
  }

  void
  QtVmdGlWindow::resizeGl (int width, int height)
  {
    m_displayDevice->resize_window (width, height);
  }

  void
  QtVmdGlWindow::keyPressEvent (QKeyEvent* event)
  {
    if (m_respondToEvents)
    {
      m_displayDevice->m_lastEvent = event->type ();
      m_displayDevice->m_lastButton = event->key ();
      m_displayDevice->m_keyboardModifiers = event->modifiers ();
      event->accept ();
    }
    else
      event->ignore ();
  }

  void
  QtVmdGlWindow::keyReleaseEvent (QKeyEvent* event)
  {
    event->ignore ();
  }

  void
  QtVmdGlWindow::mousePressEvent (QMouseEvent* event)
  {
    if (m_respondToEvents)
    {
      m_displayDevice->m_lastEvent = event->type ();
      m_displayDevice->m_lastButton = event->button ();
      m_displayDevice->m_keyboardModifiers = event->modifiers ();
      m_displayDevice->m_lastMouseX = event->x ();
      m_displayDevice->m_lastMouseY = event->y ();
      event->accept ();
    }
    else
      event->ignore ();
  }

  void
  QtVmdGlWindow::mouseReleaseEvent (QMouseEvent* event)
  {
    if (m_respondToEvents)
    {
      m_displayDevice->m_lastEvent = event->type ();
      m_displayDevice->m_lastButton = event->button ();
      m_displayDevice->m_keyboardModifiers = event->modifiers ();
      m_displayDevice->m_lastMouseX = event->x ();
      m_displayDevice->m_lastMouseY = event->y ();
      event->accept ();
    }
    else
      event->ignore ();
  }

  void
  QtVmdGlWindow::mouseMoveEvent (QMouseEvent* event)
  {
    m_displayDevice->m_keyboardModifiers = event->modifiers ();
    m_displayDevice->m_lastMouseX = event->x ();
    m_displayDevice->m_lastMouseY = event->y ();
    event->ignore ();
  }

  void
  QtVmdGlWindow::wheelEvent (QWheelEvent* event)
  {
    if (m_respondToEvents)
    {
      m_displayDevice->m_lastEvent = event->type ();
      m_displayDevice->m_lastZDelta = event->delta ();
      event->accept ();
    }
    else
      event->ignore ();
  }

}
