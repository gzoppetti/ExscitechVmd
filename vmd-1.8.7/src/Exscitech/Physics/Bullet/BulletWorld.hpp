#ifndef BULLETWORLD_HPP_
#define BULLETWORLD_HPP_

#include <vector>

#include <bullet/btBulletDynamicsCommon.h>
#include <bullet/btBulletCollisionCommon.h>

#include "Exscitech/Utilities/BulletUtility.hpp"
#include "Exscitech/Utilities/BulletGlDebugDraw.hpp"

namespace Exscitech
{

  class BulletWorld
  {
  public:

    BulletWorld () :
        m_debugMode (false)
    {
      //m_broadphase = new btDbvtBroadphase ();
      m_broadphase = new btAxisSweep3 (btVector3 (-50, -50, -50),
          btVector3 (50, 50, 50), 5000);

      m_collisionConfiguration = new btDefaultCollisionConfiguration ();

      m_dispatcher = new btCollisionDispatcher (m_collisionConfiguration);

      m_solver = new btSequentialImpulseConstraintSolver ();

      m_dynamicsWorld = new btDiscreteDynamicsWorld (m_dispatcher, m_broadphase,
          m_solver, m_collisionConfiguration);

      BulletUtility::clearShapeMap ();
    }

    ~BulletWorld ()
    {
      for (int i = 0; i < m_dynamicsWorld->getNumCollisionObjects (); ++i)
      {
        btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray ()[i];
        m_dynamicsWorld->removeCollisionObject (obj);
        delete obj;
      }
      delete m_dynamicsWorld;
      delete m_broadphase;
      delete m_collisionConfiguration;
      delete m_dispatcher;
      delete m_solver;

    }

    btDiscreteDynamicsWorld*
    getBulletCollisionWorld ()
    {
      return m_dynamicsWorld;
    }

    void
    setGravity (const Vector3f& gravity)
    {
      m_dynamicsWorld->setGravity (btVector3 (gravity.x, gravity.y, gravity.z));
    }

    void
    addRigidBodyToWorld (btRigidBody* body)
    {
      m_dynamicsWorld->addRigidBody (body);
    }

    void
    addRigidBodyToWorld (btRigidBody* body, short int group, short int mask)
    {
      m_dynamicsWorld->addRigidBody (body, group, mask);
    }

    void
    removeRigidBody (btRigidBody* body)
    {
      m_dynamicsWorld->removeRigidBody (body);
    }

    void
    addCollisionObjectToWorld (btCollisionObject* object)
    {
      m_dynamicsWorld->addCollisionObject (object);
    }

    void
    addCollisionObjectToWorld (btCollisionObject* object, short int group,
        short int mask)
    {
      m_dynamicsWorld->addCollisionObject (object, group, mask);
    }

    void
    addConstraint (btTypedConstraint* constraint,
        bool disableCollisions = false)
    {
      m_dynamicsWorld->addConstraint (constraint, disableCollisions);
    }

    void
    stepSimulation (float timestepInSeconds)
    {
      m_dynamicsWorld->stepSimulation (timestepInSeconds, 10.0f);

    }

    void
    debugOn ()
    {
      m_debugMode = true;

      BulletGlDebugDraw* debug = new BulletGlDebugDraw ();
      debug->setDebugMode (
          btIDebugDraw::DBG_DrawWireframe | btIDebugDraw::DBG_DisableBulletLCP);

      m_dynamicsWorld->setDebugDrawer (debug);

    }
    void
    drawDebugMode ()
    {
      if (m_debugMode)
        m_dynamicsWorld->debugDrawWorld ();
    }

  private:
    btBroadphaseInterface* m_broadphase;
    btDefaultCollisionConfiguration* m_collisionConfiguration;
    btCollisionDispatcher* m_dispatcher;
    btSequentialImpulseConstraintSolver* m_solver;
    btDiscreteDynamicsWorld* m_dynamicsWorld;

    bool m_debugMode;
  };
}
#endif
