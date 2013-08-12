#ifndef TRANSFORM_UTILITY_HPP_
#define TRANSFORM_UTILITY_HPP_

#include "Displayable.h"

#include "Exscitech/Graphics/Animation/KeyFrame.hpp"

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Quaternion.hpp"
#include "Exscitech/Math/Matrix3x3.hpp"
#include "Exscitech/Math/Matrix3x4.hpp"

namespace Exscitech
{
  class Transformable;

  class TransformUtility
  {
  public:

    static Vector3f
    getRight (const Displayable* displayable);

    static Vector3f
    getUp (const Displayable* displayable);

    static Vector3f
    getBack (const Displayable* displayable);

    static Vector3f
    getPosition (const Displayable* displayable);

    static Matrix3x4f
    getTransform(const Displayable* displayable);

    static void
    setPosition (Displayable* displayable, const Vector3f& position);

    static void
    setPosition (Displayable* displayable, const Transformable* transformable);

    static void
    setTransform (Displayable* displayable, const Vector3f& position,
        const Quaternion& orientation);

    static void
    setTransform (Displayable* displayable, const Transformable* transformable);

    static void
    setTransform (Displayable* displayable, const KeyFrame& keyFrame);

    static void
    moveForward (Displayable* displayable, float distance);

    static void
    moveBackward (Displayable* displayable, float distance);

    static void
    moveRight (Displayable* displayable, float distance);

    static void
    moveLeft (Displayable* displayable, float distance);

    static void
    moveUp (Displayable* displayable, float distance);

    static void
    moveDown (Displayable* displayable, float distance);

    static void
    moveLocal (Displayable* displayable, const Vector3f& direction,
        float distance);

    static void
    moveWorld (Displayable* displayable, const Vector3f& direction,
        float distance);

    static void
    pitch (Displayable* displayable, float angleDegrees);

    static void
    yaw (Displayable* displayable, float angleDegrees);

    static void
    roll (Displayable* displayable, float angleDegrees);

    static void
    rotateAboutPoint (Displayable* displayable, const float angleDegrees,
        const Vector3f& axis, const Vector3f& point);

    static void
    setRotation (Displayable* displayable, const Transformable* transformable);

    static void
    setRotation (Displayable* displayable, const Matrix3x3f& rotation);

    void
    setRotation (Displayable* displayable, const Quaternion& orientation);

  private:

    TransformUtility ();

  };

}

#endif
