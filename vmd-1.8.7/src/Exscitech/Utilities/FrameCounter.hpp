#ifndef FRAMECOUNTER_HPP_
#define FRAMECOUNTER_HPP_

#include <sstream>
#include <cstdio>

#include <QtOpenGL/QGLWidget>

#include "Exscitech/Games/GameController.hpp"

namespace Exscitech
{
  class FrameCounter
  {
  public:

    FrameCounter (QGLWidget* window, int x, int y) :
        m_window (window), m_font ("Courier", 20, QFont::Light, true), m_x (x), m_y (
            y), m_currentFrameCount (0), m_previousFrameCount ("??"), m_timeDelta (
            0.00f)
    {
    }

    void
    setPosition (int x, int y)
    {
      m_x = x;
      m_y = y;
    }

    void
    setWindow (QGLWidget* window)
    {
      m_window = window;
    }

    void
    update (float deltaSeconds)
    {
      m_timeDelta += deltaSeconds;

      if (m_timeDelta > 1.0f)
      {
        m_timeDelta = std::fmod(m_timeDelta, 1.0f);
        std::stringstream stream;
        stream << m_currentFrameCount;
        m_previousFrameCount = stream.str ();
        m_currentFrameCount = 1;
      }
      else
      {
        m_currentFrameCount++;
      }
    }

    void
    draw ()
    {
      glActiveTexture(GL_TEXTURE0);
      GameController::m_vmdGlWindow->renderText (m_x, m_y,
          m_previousFrameCount.c_str (), m_font);
    }

  private:

    QGLWidget* m_window;
    QFont m_font;

    int m_x;
    int m_y;
    int m_currentFrameCount;
    std::string m_previousFrameCount;
    float m_timeDelta;
  };
}

#endif
