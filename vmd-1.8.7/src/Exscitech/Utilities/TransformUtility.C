#include "Molecule.h"
#include "Matrix4.h"

#include "Exscitech/Graphics/Transformable.hpp"

#include "Exscitech/Math/Matrix3x3.hpp"

#include "Exscitech/Utilities/TransformUtility.hpp"

namespace Exscitech
{
  Vector3f
  TransformUtility::getRight (const Displayable* displayable)
  {
    Vector3f right (&displayable->rotm.mat[0]);
    return (right);
  }

  Vector3f
  TransformUtility::getUp (const Displayable* displayable)
  {
    Vector3f up (&displayable->rotm.mat[4]);
    return (up);
  }

  Vector3f
  TransformUtility::getBack (const Displayable* displayable)
  {
    Vector3f back (&displayable->rotm.mat[8]);
    return (back);
  }

  Vector3f
  TransformUtility::getPosition (const Displayable* displayable)
  {
    Vector3f translation (displayable->globt);
    return (translation);
  }

  Matrix3x4f
  TransformUtility::getTransform (const Displayable* displayable)
  {
    float scale = displayable->scale;
    Vector3f globalTranslation (displayable->globt);
    Vector3f centeringTranslation (displayable->centt);
    Matrix3x4f transform (displayable->rotm.mat);

    transform.rotation *= scale;
    transform.translation = transform.rotation * globalTranslation + centeringTranslation;

    return transform;
  }

  void
  TransformUtility::setPosition (Displayable* displayable,
      const Vector3f& position)
  {
    displayable->set_glob_trans (position.x, position.y, position.z);
  }

  void
  TransformUtility::setTransform (Displayable* displayable,
      const Vector3f& position, const Quaternion& orientation)
  {
    setPosition (displayable, position);
    float orientationArray[16];
    orientation.getAsArray (orientationArray);
    Matrix4 vmdRotationMatrix (orientationArray);
    displayable->set_rot (vmdRotationMatrix);
  }

  void
  TransformUtility::setTransform (Displayable* displayable,
      const KeyFrame& keyFrame)
  {
    setTransform (displayable, keyFrame.getPosition (),
        keyFrame.getOrientation ());
  }

  void
  TransformUtility::moveForward (Displayable* displayable, float distance)
  {
    Vector3f forward = getBack (displayable) * -distance;
    displayable->add_glob_trans (forward.x, forward.y, forward.z);
  }

  void
  TransformUtility::moveBackward (Displayable* displayable, float distance)
  {
    Vector3f back = getBack (displayable) * distance;
    displayable->add_glob_trans (back.x, back.y, back.z);
  }

  void
  TransformUtility::moveRight (Displayable* displayable, float distance)
  {
    Vector3f right = getRight (displayable) * distance;
    displayable->add_glob_trans (right.x, right.y, right.z);
  }

  void
  TransformUtility::moveLeft (Displayable* displayable, float distance)
  {
    Vector3f left = getRight (displayable) * -distance;
    displayable->add_glob_trans (left.x, left.y, left.z);
  }

  void
  TransformUtility::moveUp (Displayable* displayable, float distance)
  {
    Vector3f up = getUp (displayable) * distance;
    displayable->add_glob_trans (up.x, up.y, up.z);
  }

  void
  TransformUtility::moveDown (Displayable* displayable, float distance)
  {
    Vector3f down = getUp (displayable) * -distance;
    displayable->add_glob_trans (down.x, down.y, down.z);
  }

  void
  TransformUtility::moveLocal (Displayable* displayable,
      const Vector3f& direction, float distance)
  {
    Vector3f right = getRight (displayable);
    Vector3f up = getUp (displayable);
    Vector3f back = getBack (displayable);
    Vector3f worldDirection = right * direction.x + up * direction.y
        + back * direction.z;
    moveWorld (displayable, worldDirection, distance);
  }

  void
  TransformUtility::moveWorld (Displayable* displayable,
      const Vector3f& direction, float distance)
  {
    displayable->add_glob_trans (direction.x * distance, direction.y * distance,
        direction.z * distance);
  }

  void
  TransformUtility::pitch (Displayable* displayable, float angleDegrees)
  {
    Matrix4 rotation;
    rotation.rot (angleDegrees, 'x');
    displayable->add_rot (rotation);
  }

  void
  TransformUtility::yaw (Displayable* displayable, float angleDegrees)
  {
    Matrix4 rotation;
    rotation.rot (angleDegrees, 'y');
    displayable->add_rot (rotation);
  }

  void
  TransformUtility::roll (Displayable* displayable, float angleDegrees)
  {
    Matrix4 rotation;
    rotation.rot (angleDegrees, 'z');
    displayable->add_rot (rotation);
  }

  void
  TransformUtility::rotateAboutPoint (Displayable* displayable,
      const float angleDegrees, const Vector3f& axis, const Vector3f& point)
  {
    Transformable temp;
    temp.rotateAboutPoint (angleDegrees, axis, point);
    float rotation[16];
    temp.getTransform (rotation);

    Matrix4 mat (rotation);
    displayable->add_rot (mat);

    Vector3f position = temp.getPosition ();
    displayable->set_glob_trans (position.x, position.y, position.z);
  }

  void
  TransformUtility::setPosition (Displayable* displayable,
      const Transformable* transformable)
  {
    Vector3f position = transformable->getPosition ();
    setPosition (displayable, position);
  }

  void
  TransformUtility::setRotation (Displayable* displayable,
      const Transformable* transformable)
  {
    Matrix3x3f rotation = transformable->getRotation ();
    setRotation (displayable, rotation);
  }

  void
  TransformUtility::setRotation (Displayable* displayable,
      const Matrix3x3f& rotation)
  {
    float rotationArray[16];
    rotation.getAsArray (rotationArray);
    Matrix4 vmdRotationMatrix (rotationArray);
    displayable->set_rot (vmdRotationMatrix);
  }

  void
  TransformUtility::setRotation (Displayable* displayable,
      const Quaternion& orientation)
  {
    float orientationArray[16];
    orientation.getAsArray (orientationArray);
    Matrix4 vmdRotationMatrix (orientationArray);
    displayable->set_rot (vmdRotationMatrix);
  }

  void
  TransformUtility::setTransform (Displayable* displayable,
      const Transformable* transformable)
  {
    Vector3f position = transformable->getPosition ();
    setPosition (displayable, position);
  }
}
