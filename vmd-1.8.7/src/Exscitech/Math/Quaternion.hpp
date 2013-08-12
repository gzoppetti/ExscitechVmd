#ifndef QUATERNION_HPP_
#define QUATERNION_HPP_

#include <iostream>

#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
  template<typename T>
    class Matrix3x3;

  // Primarily intended for unit quaternions of form q = (w, v)
  //   = (sin (angle/2), cos (angle/2) * axis)
  //   where |axis| = 1
  class Quaternion
  {
  public:

    Quaternion ();

    Quaternion (float px, float py, float pz);

    Quaternion (float pw, float px, float py, float pz);

    explicit
    Quaternion (const Vector3<float>& v);

    Quaternion (float angleDegrees, const Vector3<float>& unitAxis);

    void
    setToRotationY (float angleDegrees);

    void
    setToRotationX (float angleDegrees);

    void
    setToRotationZ (float angleDegrees);

    void
    setFromYawPitchRoll (float yawDegrees, float pitchDegrees,
        float rollDegrees);
    void
    setToRotation (float angleDegrees, const Vector3f& axis);

    void
    setFromAngleAxis (float angleDegrees, const Vector3<float>& unitAxis);

    void
    setFromMatrix (const float* const matrix4x4);

    float
    getAngleFromUnit () const;

    Vector3f
    getAxisFromUnit () const;

    // Getting right, up, and backward are somewhat costly for quaternions
    Vector3f
    getRight () const;

    Vector3f
    getUp () const;

    Vector3f
    getBackward () const;

    void
    getAsArray (float array[16]) const;

    Matrix3x3<float>
    toMatrix () const;

    void
    setToIdentity ();

    float
    dot (const Quaternion& q) const;

    float
    lengthSquared () const;

    float
    length () const;

    void
    orthonormalize ();

    float
    normalize ();

    // Optimized invert for unit quaternions
    void
    invertForUnit ();

    // General invert
    void
    invert ();

    void
    conjugate ();

    // Follow this rotation by "q"
    void
    concatenate (const Quaternion& q);

    // "destinationWeight" must be in [0, 1]
    // Result will not be unit so normalize if desired
    Quaternion
    lerp (const Quaternion& destination, float destinationWeight) const;

    // Test and optimize
    void
    slerp (const Quaternion& destination, float destinationWeight);

    // Test and optimize
    Vector3<float>
    rotatePoint (const Vector3<float>& point) const;

    // Operators
    Quaternion&
    operator+= (const Quaternion& q);

    Quaternion&
    operator-= (const Quaternion& q);

    Quaternion&
    operator*= (const Quaternion& q);

    Quaternion&
    operator*= (float scalar);

    Quaternion&
    operator/= (float scalar);

  public:

    union
    {
      float coords[4];
      struct
      {
        float w, x, y, z;
      };
    };

  };

  inline Quaternion
  operator+ (const Quaternion& q1, const Quaternion& q2)
  {
    Quaternion sum (q1);
    sum += q2;
    return (sum);
  }

  inline Quaternion
  operator- (const Quaternion& q1, const Quaternion& q2)
  {
    Quaternion diff (q1);
    diff -= q2;
    return (diff);
  }

  inline Quaternion
  operator* (const Quaternion& q1, const Quaternion& q2)
  {
    Quaternion prod (q1);
    prod *= q2;
    return (prod);
  }

  inline Quaternion
  operator* (const Quaternion& q, float scalar)
  {
    Quaternion multiple (q);
    multiple *= scalar;
    return (multiple);
  }

  inline Quaternion
  operator* (float scalar, const Quaternion& q)
  {
    return (q * scalar);
  }

  inline Quaternion
  operator/ (const Quaternion& q, float scalar)
  {
    Quaternion quotient (q);
    quotient /= scalar;
    return (quotient);
  }
}

#endif
