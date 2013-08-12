#ifndef BULLETCALLBACKS_HPP_
#define BULLETCALLBACKS_HPP_

#include <BulletDynamics/Dynamics/btRigidBody.h>
#include <btBulletDynamicsCommon.h>

namespace Exscitech
{
  class ClosestNotMeRayResultCallback : public btCollisionWorld::ClosestRayResultCallback
  {
  public:

    ClosestNotMeRayResultCallback (btCollisionObject* me, btVector3 from,
        btVector3 to) :
        btCollisionWorld::ClosestRayResultCallback (from, to)
    {
      m_me = me;
    }

    virtual btScalar
    addSingleResult (btCollisionWorld::LocalRayResult& rayResult,
        bool normalInWorldSpace)
    {
      if (rayResult.m_collisionObject == m_me)
        return 1.0;

      return ClosestRayResultCallback::addSingleResult (rayResult,
          normalInWorldSpace);
    }

  protected:
    btCollisionObject* m_me;

  };
}
#endif
