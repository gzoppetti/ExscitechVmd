#ifndef TRANSFORMABLE_HPP_
#define TRANSFORMABLE_HPP_

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Matrix3x3.hpp"
#include "Exscitech/Math/Matrix3x4.hpp"
#include "Exscitech/Math/Matrix4x4.hpp"
namespace Exscitech
{

  class Transformable
  {
  public:

    Transformable ();

    void
    setToIdentity ();

    void
    reset ();

    void
    setToZero ();

    void
    setPosition (const Vector3f& position);

    void
    setTransform (const float* const matrix4x4);

    void
    setTranslationAndOrientation (const float* const matrix4x4);

    void
    setTranslationAndOrientation (const Matrix3x4f& transform);

    void
    setTranslation (const float* const matrix4x4);

    void
    setTransform (const Matrix3x4f& transform);

    Vector3f
    getRight () const;

    Vector3f
    getUp () const;

    Vector3f
    getBackward () const;

    Vector3f
    getForward () const;

    Vector3f
    getTranslation () const;

    Vector3f
    getPosition () const;

    Matrix3x4f
    getTransform () const;

    Matrix4x4f
    getTransform4x4() const;

    void
    getTransform (float* const arrayOf16) const;

    void
    getTransformWithoutScale (float array[16]) const;

    Matrix3x4f
    getTransformWithoutScale () const;

    Matrix3x3f
    getRotation () const;

    Matrix3x4f
    getRotation3x4 () const;

    Matrix3x3f
    getRotationAndScale () const;

    Vector3f
    getScale () const;

    void
    setRotationAndScale (const Matrix3x3f& rotationAndScale);

    void
    setRotation (const Matrix3x3f& rotation);

    void
    moveForward (float distance);

    void
    moveBackward (float distance);

    void
    moveRight (float distance);

    void
    moveLeft (float distance);

    void
    moveUp (float distance);

    void
    moveDown (float distance);

    void
    moveLocal (const Vector3f& direction, float distance);

    void
    lerpPosition (const Vector3f& destination, float destinationWeight);

    void
    moveWorld (const Vector3f& direction, float distance);

    // Rotate about the local Y axis
    void
    yaw (float angleDegrees);
    // Rotate about the local X axis
    void
    pitch (float angleDegrees);
    // Rotate about the local Z axis

    void
    roll (float angleDegrees);
    // Rotate about an arbitrary axis
    void
    rotateLocal (float angleDegrees, const Vector3f& axis);

    void
    alignRightWithVector (const Vector3f& direction);

    void
    alignUpWithVector (const Vector3f& direction);

    void
    alignBackwardWithVector (const Vector3f& direction);

    void
    alignWithWorldY ();

    void
    rotateWorld (float angleDegrees, const Vector3f& axis);

    void
    rotateWorldYInPlace (float angleDegrees);

    void
    rotateAboutPoint (float angleDegrees, const Vector3f& axis,
        const Vector3f& point);

    void
    scaleX (float scale);

    void
    scaleY (float scale);

    void
    scaleZ (float scale);

    void
    scaleLocal (float scale);

    void
    scaleLocal (const Vector3f& scales);

    void
    scaleWorld (float scale);

    void
    orthonormalize ();

  private:

    Vector3f m_right;
    Vector3f m_up;
    Vector3f m_backward;
    Vector3f m_translation;
    Vector3f m_scales;

  };
}
#endif
