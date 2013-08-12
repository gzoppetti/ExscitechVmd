#include "Exscitech/Constants.hpp"

#include "Exscitech/Graphics/Transformable.hpp"

#include "Exscitech/Math/Matrix3x3.hpp"
#include "Exscitech/Math/Matrix3x4.hpp"
#include "Exscitech/Math/Math.hpp"

namespace Exscitech
{
  Transformable::Transformable () :
      m_right (1, 0, 0), m_up (0, 1, 0), m_backward (0, 0, 1), m_translation (
          0.0f), m_scales (1.0f)
  {
  }

  void
  Transformable::setToIdentity ()
  {
    m_right = Constants::WORLD_X;
    m_up = Constants::WORLD_Y;
    m_backward = Constants::WORLD_Z;
    m_translation.setToZero ();
    m_scales.setToOne ();
  }

  void
  Transformable::reset ()
  {
    setToIdentity ();
  }

  void
  Transformable::setToZero ()
  {
    m_right.setToZero ();
    m_up.setToZero ();
    m_backward.setToZero ();
    m_translation.setToZero ();
    m_scales.setToZero ();
  }

  void
  Transformable::setPosition (const Vector3f& position)
  {
    m_translation = position;
  }

  void
  Transformable::setTransform (const float* const matrix4x4)
  {
    m_right.set (*(matrix4x4 + 0), *(matrix4x4 + 1), *(matrix4x4 + 2));
    m_scales.x = m_right.normalize ();
    m_up.set (*(matrix4x4 + 4), *(matrix4x4 + 5), *(matrix4x4 + 6));
    m_scales.y = m_up.normalize ();
    m_backward.set (*(matrix4x4 + 8), *(matrix4x4 + 9), *(matrix4x4 + 10));
    m_scales.z = m_backward.normalize ();
    m_translation.set (*(matrix4x4 + 12), *(matrix4x4 + 13), *(matrix4x4 + 14));
  }

  void
  Transformable::setTranslationAndOrientation (const float* const matrix4x4)
  {
    m_right.set (*(matrix4x4 + 0), *(matrix4x4 + 1), *(matrix4x4 + 2));
    m_right.normalize ();
    m_up.set (*(matrix4x4 + 4), *(matrix4x4 + 5), *(matrix4x4 + 6));
    m_up.normalize ();
    m_backward.set (*(matrix4x4 + 8), *(matrix4x4 + 9), *(matrix4x4 + 10));
    m_backward.normalize ();
    m_translation.set (*(matrix4x4 + 12), *(matrix4x4 + 13), *(matrix4x4 + 14));
  }

  void
  Transformable::setTranslationAndOrientation (const Matrix3x4f& transform)
  {
    m_right = transform.rotation.getRight ();
    m_right.normalize ();
    m_up = transform.rotation.getUp ();
    m_up.normalize ();
    m_backward = transform.rotation.getBackward ();
    m_backward.normalize ();
    m_translation = transform.translation;
  }

  void
  Transformable::setTranslation (const float* const matrix4x4)
  {
    m_translation.set (*(matrix4x4 + 12), *(matrix4x4 + 13), *(matrix4x4 + 14));
  }

  void
  Transformable::setTransform (const Matrix3x4f& transform)
  {
    m_right = transform.rotation.getRight ();
    m_scales.x = m_right.normalize ();
    m_up = transform.rotation.getUp ();
    m_scales.y = m_up.normalize ();
    m_backward = transform.rotation.getBackward ();
    m_scales.z = m_backward.normalize ();
    m_translation = transform.translation;
  }

  Vector3f
  Transformable::getRight () const
  {
    return (m_right);
  }

  Vector3f
  Transformable::getUp () const
  {
    return (m_up);
  }

  Vector3f
  Transformable::getBackward () const
  {
    return (m_backward);
  }

  Vector3f
  Transformable::getForward () const
  {
    return (-m_backward);
  }

  Vector3f
  Transformable::getTranslation () const
  {
    return (m_translation);
  }

  Vector3f
  Transformable::getPosition () const
  {
    return (m_translation);
  }

  Matrix3x4f
  Transformable::getTransform () const
  {
    return (Matrix3x4f (m_right * m_scales.x, m_up * m_scales.y,
        m_backward * m_scales.z, m_translation));
  }

  Matrix4x4f
  Transformable::getTransform4x4 () const
  {
    return Matrix4x4f (m_right * m_scales.x, m_up * m_scales.y,
        m_backward * m_scales.z, m_translation);
  }

  void
  Transformable::getTransform (float* const arrayOf16) const
  {
    Matrix3x4f transform (m_right * m_scales.x, m_up * m_scales.y,
        m_backward * m_scales.z, m_translation);
    transform.getAsArray (arrayOf16);
  }

  void
  Transformable::getTransformWithoutScale (float array[16]) const
  {
    Matrix3x4f transform (m_right, m_up, m_backward, m_translation);
    transform.getAsArray (array);
  }

  Matrix3x4f
  Transformable::getTransformWithoutScale () const
  {
    return Matrix3x4f (m_right, m_up, m_backward, m_translation);
  }

  Matrix3x3f
  Transformable::getRotation () const
  {
    return (Matrix3x3f (m_right, m_up, m_backward));
  }

  Matrix3x4f
  Transformable::getRotation3x4 () const
  {

    return (Matrix3x4f (m_right, m_up, m_backward));
  }
  Matrix3x3f
  Transformable::getRotationAndScale () const
  {
    return (Matrix3x3f (m_right * m_scales.x, m_up * m_scales.y,
        m_backward * m_scales.z));
  }

  Vector3f
  Transformable::getScale () const
  {
    return m_scales;
  }

  void
  Transformable::setRotationAndScale (const Matrix3x3f& rotationAndScale)
  {
    m_right = rotationAndScale.getRight ();
    m_scales.x = m_right.normalize ();
    m_up = rotationAndScale.getUp ();
    m_scales.y = m_up.normalize ();
    m_backward = rotationAndScale.getBackward ();
    m_scales.z = m_backward.normalize ();
  }

  void
  Transformable::setRotation (const Matrix3x3f& rotation)
  {
    m_right = rotation.getRight ();
    m_right.normalize ();
    m_up = rotation.getUp ();
    m_up.normalize ();
    m_backward = rotation.getBackward ();
    m_backward.normalize ();
  }

  void
  Transformable::moveForward (float distance)
  {
    Vector3f forward = distance * -m_backward;
    m_translation += forward;
  }

  void
  Transformable::moveBackward (float distance)
  {
    Vector3f backward = distance * m_backward;
    m_translation += backward;
  }

  void
  Transformable::moveRight (float distance)
  {
    Vector3f right = distance * m_right;
    m_translation += right;
  }

  void
  Transformable::moveLeft (float distance)
  {
    Vector3f left = distance * -m_right;
    m_translation += left;
  }

  void
  Transformable::moveUp (float distance)
  {
    Vector3f up = distance * m_up;
    m_translation += up;
  }

  void
  Transformable::moveDown (float distance)
  {
    Vector3f down = distance * -m_up;
    m_translation += down;
  }

  void
  Transformable::moveLocal (const Vector3f& direction, float distance)
  {
    Vector3f worldDirection = direction.x * m_right + direction.y * m_up
        + direction.z * m_backward;
    moveWorld (worldDirection, distance);
  }

  void
  Transformable::lerpPosition (const Vector3f& destination,
      float destinationWeight)
  {
    m_translation = m_translation.lerp (destination, destinationWeight);
  }

  void
  Transformable::moveWorld (const Vector3f& direction, float distance)
  {
    m_translation += distance * direction;
  }

  void
  Transformable::yaw (float angleDegrees)
  {
    Matrix3x3f rotation;
    rotation.setFromAngleAxis (angleDegrees, m_up);
    m_right = rotation * m_right;
    m_backward = rotation * m_backward;
    orthonormalize ();
  }

  void
  Transformable::pitch (float angleDegrees)
  {
    Matrix3x3f rotation;
    rotation.setFromAngleAxis (angleDegrees, m_right);
    m_up = rotation * m_up;
    m_backward = rotation * m_backward;
    orthonormalize ();
  }

  void
  Transformable::roll (float angleDegrees)
  {
    Matrix3x3f rotation;
    rotation.setFromAngleAxis (angleDegrees, m_backward);
    m_right = rotation * m_right;
    m_up = rotation * m_up;
    orthonormalize ();
  }

  void
  Transformable::rotateLocal (float angleDegrees, const Vector3f& axis)
  {
    Matrix3x3f rotation;
    rotation.setFromAngleAxis (angleDegrees, axis);
    Matrix3x3f orientation = getRotationAndScale () * rotation;
    m_right = orientation.getRight ();
    m_up = orientation.getUp ();
    m_backward = orientation.getBackward ();
    orthonormalize ();
  }

  void
  Transformable::alignWithWorldY ()
  {
    m_up = Constants::WORLD_Y;
    if (m_backward.x != 0 || m_backward.z != 0)
    {
      m_backward.y = 0;
      m_backward.normalize ();
    }
    else
    {
      // Was looking along Y or -Y
      m_backward.set (0, 0, 1);
    }
    m_right = m_up.cross (m_backward);
  }

  void
  Transformable::alignRightWithVector (const Vector3f& direction)
  {
    m_right = direction;
    Vector3f up = m_backward.cross (m_right);
    if (up != Constants::ZERO)
    {
      // Generate new up
      m_up = up;
      m_up.normalize ();
    }
    m_backward = m_right.cross (m_up);
  }

  void
  Transformable::alignUpWithVector (const Vector3f& direction)
  {
    m_up = direction;
    Vector3f backward = m_right.cross (m_up);
    if (backward != Constants::ZERO)
    {
      // Generate new backward
      m_backward = backward;
      m_backward.normalize ();
    }
    m_right = m_up.cross (m_backward);

    //  Alternative implementation suitable for quaternions
    //
    //  Single angleBetweenRad = computeAngleBetween (m_up, direction, true);
    //  Single angleBetween = Math::toDegrees (angleBetweenRad);
    //  Vector3f rotationAxis;
    //  if (!Math::equalsUsingTolerance (angleBetween, M_PI))
    //  {
    //    rotationAxis = m_up.cross (direction);
    //    rotationAxis.normalize ();
    //  }
    //  else
    //  {
    //    rotationAxis = m_right;
    //  }
    //  Matrix3x3f rotation;
    //  rotation.setFromAngleAxis (angleBetween, rotationAxis);
    //  Matrix3x3f thisRotation = getRotationAndScale ();
    //  thisRotation *= rotation;
    //  setRotationAndScale (thisRotation);
  }

  void
  Transformable::alignBackwardWithVector (const Vector3f& direction)
  {
    m_backward = direction;
    Vector3f right = m_up.cross (m_backward);
    if (right != Constants::ZERO)
    {
      // Generate new right
      m_right = right;
      m_right.normalize ();
    }
    m_up = m_backward.cross (m_right);
  }

  void
  Transformable::rotateWorld (float angleDegrees, const Vector3f& axis)
  {
    Matrix3x3f rotation;
    rotation.setFromAngleAxis (angleDegrees, axis);
    m_right = rotation * m_right;
    m_up = rotation * m_up;
    m_backward = rotation * m_backward;
    m_translation = rotation * m_translation;
    orthonormalize ();
  }

  void
  Transformable::rotateWorldYInPlace (float angleDegrees)
  {
    Matrix3x3f rotation;
    rotation.setToRotationY (angleDegrees);
    m_right = rotation * m_right;
    m_up = rotation * m_up;
    m_backward = rotation * m_backward;
    orthonormalize ();
  }

  void
  Transformable::rotateAboutPoint (float angleDegrees, const Vector3f& axis,
      const Vector3f& point)
  {
    Matrix3x3f rotation;
    rotation.setFromAngleAxis (angleDegrees, axis);
    m_right = rotation * m_right;
    m_up = rotation * m_up;
    m_backward = rotation * m_backward;
    m_translation = rotation * m_translation;
    Vector3f negatedPointRot = rotation * -point;
    m_translation += negatedPointRot + point;
    orthonormalize ();
  }

  void
  Transformable::scaleX (float scale)
  {
    m_scales.x *= scale;
  }

  void
  Transformable::scaleY (float scale)
  {
    m_scales.y *= scale;
  }

  void
  Transformable::scaleZ (float scale)
  {
    m_scales.z *= scale;
  }

  void
  Transformable::scaleLocal (float scale)
  {
    m_scales *= scale;
  }

  void
  Transformable::scaleLocal (const Vector3f& scales)
  {
    m_scales.multiplyComponents (scales);
  }

  void
  Transformable::scaleWorld (float scale)
  {
    // Uniform scale only
    scaleLocal (scale);
    m_translation *= scale;
  }

  void
  Transformable::orthonormalize ()
  {
    m_backward.normalize ();
    m_right = m_up.cross (m_backward);
    m_right.normalize ();
    m_up = m_backward.cross (m_right);
  }

}
