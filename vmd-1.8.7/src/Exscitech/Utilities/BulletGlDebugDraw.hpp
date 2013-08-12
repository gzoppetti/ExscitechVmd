#ifndef BULLET_GL_DEBUG_DRAW_HPP_
#define BULLET_GL_DEBUG_DRAW_HPP_

#include <GL/glew.h>
#include <bullet/btBulletCollisionCommon.h>

#include "Exscitech/Utilities/CameraUtility.hpp"

namespace Exscitech
{

  class BulletGlDebugDraw : public btIDebugDraw
  {
  public:

    BulletGlDebugDraw ()
    {

    }

    virtual void
    drawLine (const btVector3& from, const btVector3& to,
        const btVector3& color)
    {
      glColor3fv (color);
      glBegin (GL_LINES);
      glVertex3fv (from);
      glVertex3fv (to);
      glEnd ();
    }

    virtual void
    drawContactPoint (const btVector3& PointOnB, const btVector3& normalOnB,
        btScalar distance, int lifeTime, const btVector3& color)
    {
    }

    virtual void
    reportErrorWarning (const char* warningString)
    {
    }

    virtual void
    draw3dText (const btVector3& location, const char* textString)
    {
    }

    virtual void
    setDebugMode (int debugMode)
    {
      m_debugMode = debugMode;
    }

    virtual int
    getDebugMode () const
    {
      return m_debugMode;
    }

  private:

    int m_debugMode;

  };
}
#endif
