
#include "Exscitech/Constants.hpp"

#include "Exscitech/Math/Quaternion.hpp"
#include "Exscitech/Math/Matrix3x3.hpp"


namespace Exscitech
{
  Quaternion::Quaternion () :
    w (1), x (0), y (0), z (0)
  {
  }

  Quaternion::Quaternion (float px, float py, float pz) :
    w (0), x (px), y (py), z (pz)
  {
  }

  Quaternion::Quaternion (float pw, float px, float py, float pz) :
    w (pw), x (px), y (py), z (pz)
  {
  }

  Quaternion::Quaternion (const Vector3<float>& v) :
    w (0), x (v.x), y (v.y), z (v.z)
  {
  }

  Quaternion::Quaternion (float angleDegrees, const Vector3<float>& unitAxis)
  {
    setFromAngleAxis (angleDegrees, unitAxis);
  }

  void
  Quaternion::setToRotationY (float angleDegrees)
  {
    setFromAngleAxis (angleDegrees, Constants::WORLD_Y);
  }

  void
  Quaternion::setToRotationX (float angleDegrees)
  {
    setFromAngleAxis (angleDegrees, Constants::WORLD_X);
  }

  void
  Quaternion::setToRotationZ (float angleDegrees)
  {
    setFromAngleAxis (angleDegrees, Constants::WORLD_Z);
  }

  void
  Quaternion::setFromYawPitchRoll (float yawDegrees, float pitchDegrees,
      float rollDegrees)
  {
    float cosYaw = cos (Math<float>::toRadians (yawDegrees) / 2);
    float sinYaw = sin (Math<float>::toRadians (yawDegrees) / 2);
    float cosPitch = cos (Math<float>::toRadians (pitchDegrees) / 2);
    float sinPitch = sin (Math<float>::toRadians (pitchDegrees) / 2);
    float cosRoll = cos (Math<float>::toRadians (rollDegrees) / 2);
    float sinRoll = sin (Math<float>::toRadians (rollDegrees) / 2);
    w = cosPitch * cosYaw * cosRoll + sinPitch * sinYaw * sinRoll;
    x = sinPitch * cosYaw * cosRoll - cosPitch * sinYaw * sinRoll;
    y = cosPitch * sinYaw * cosRoll + sinPitch * cosYaw * sinRoll;
    z = cosPitch * cosYaw * sinRoll - sinPitch * sinYaw * cosRoll;
  }

  void
  Quaternion::setFromAngleAxis (float angleDegrees,
      const Vector3<float>& unitAxis)
  {
    float angleRad = Math<float>::toRadians (angleDegrees);
    w = cos (angleRad * 0.5);
    float sinAngleOver2 = sin (angleDegrees * 0.5);
    x = sinAngleOver2 * unitAxis.x;
    y = sinAngleOver2 * unitAxis.y;
    z = sinAngleOver2 * unitAxis.z;
  }

  void
  Quaternion::setFromMatrix (const float* const matrix4x4)
  {
    Matrix3x3f rotation (matrix4x4);
    *this = rotation.toQuaternion ();
  }

  float
  Quaternion::getAngleFromUnit () const
  {
    // w = cos (angle * 0.5)
    float angleRad = acos (w) * 2;
    float angleDegrees = Math<float>::toDegrees (angleRad);
    return (angleDegrees);
  }

  Vector3f
  Quaternion::getAxisFromUnit () const
  {
    // (x, y, z) = sin (angle * 0.5) * axis
    // w^2 + |v|^2 = 1 ==> |v| = sqrt (1 - w^2)
    float invSin = 1 / (sqrt (1 - w * w));
    return (Vector3f (invSin * x, invSin * y, invSin * z));
  }

  void
  Quaternion::getAsArray (float array[16]) const
  {
    Matrix3x3f rotation = toMatrix ();
    rotation.getAsArray (array);
  }

  Matrix3x3f
  Quaternion::toMatrix () const
  {
    Matrix3x3f rotation;
    rotation.setRight (getRight ());
    rotation.setUp (getUp ());
    rotation.setBackward (getBackward ());

    return (rotation);
  }
  // Getting right, up, and backward are somewhat costly for quaternions
  Vector3f
  Quaternion::getRight () const
  {
    float rx = 1 - 2 * (y * y + z * z);
    float ry = 2 * (x * y + w * z);
    float rz = 2 * (x * z - w * y);

    return (Vector3f (rx, ry, rz));
  }

  Vector3f
  Quaternion::getUp () const
  {
    float rx = 2 * (x * y - w * z);
    float ry = 1 - 2 * (x * x + z * z);
    float rz = 2 * (y * z + w * x);

    return (Vector3f (rx, ry, rz));
  }

  Vector3f
  Quaternion::getBackward () const
  {
    float rx = 2 * (x * z + w * y);
    float ry = 2 * (y * z - w * x);
    float rz = 1 - 2 * (x * x + y * y);

    return (Vector3f (rx, ry, rz));
  }

  void
  Quaternion::setToIdentity ()
  {
    w = 1;
    x = y = z = 0;
  }

  float
  Quaternion::dot (const Quaternion& q) const
  {
    return (w * q.w + x * q.x + y * q.y + z * q.z);
  }

  float
  Quaternion::lengthSquared () const
  {
    // sqrt(q * conj(q)) = |q|
    return (dot (*this));
  }

  float
  Quaternion::length () const
  {
    // q * conj(q) = w^2 + x^2 + y^2 + z^2 = |q|^2
    return (sqrt (lengthSquared ()));
  }

  void
  Quaternion::orthonormalize ()
  {
    normalize ();
  }

  float
  Quaternion::normalize ()
  {
    float lLength = length ();
    float invLength = 1 / lLength;
    w *= invLength;
    x *= invLength;
    y *= invLength;
    z *= invLength;
    return (lLength);
  }

  // Optimized invert for unit quaternions
  void
  Quaternion::invertForUnit ()
  {
    conjugate ();
  }

  // General invert
  void
  Quaternion::invert ()
  {
    // q * conj(q) = |q|^2
    //   ==> q * conj(q) / |q|^2 = 1
    //   ==> inv(q) = conj(q) / |q|^2
    float invLengthSq = 1 / lengthSquared ();
    w *= +invLengthSq;
    x *= -invLengthSq;
    y *= -invLengthSq;
    z *= -invLengthSq;
  }

  void
  Quaternion::conjugate ()
  {
    // q * conj(q) = (w + v)(w - v) = w^2 - v*v
    //             = w^2 + x^2 + y^2 + z^2 + 0v
    x = -x;
    y = -y;
    z = -z;
  }

  // Follow this rotation by "q"
  void
  Quaternion::concatenate (const Quaternion& q)
  {
    // q * this * p * this' * q'
    *this = q * *this;
  }

  // "destinationWeight" must be in [0, 1]
  // Result will not be unit so normalize if desired
  Quaternion
  Quaternion::lerp (const Quaternion& destination, float destinationWeight) const
  {
    float srcWeight = 1 - destinationWeight;
    float wr = srcWeight * w + destinationWeight * destination.w;
    float xr = srcWeight * x + destinationWeight * destination.x;
    float yr = srcWeight * y + destinationWeight * destination.y;
    float zr = srcWeight * z + destinationWeight * destination.z;
    return (Quaternion (wr, xr, yr, zr));
  }

  // Test and optimize
  void
  Quaternion::slerp (const Quaternion& destination, float destinationWeight)
  {
    float lDot = dot (destination);
    float sinTheta = sqrt (1 - lDot * lDot);
    float theta = asin (sinTheta);
    float thisFactor = sin (theta * (1 - destinationWeight)) / sinTheta;
    float destinationFactor = sin (theta * destinationWeight) / sinTheta;
    *this = thisFactor * *this + destinationFactor * destination;
  }

  // Test and optimize
  Vector3<float>
  Quaternion::rotatePoint (const Vector3<float>& point) const
  {
    // p' = q * p * q-1
    Quaternion thisInverse (*this);
    thisInverse.invert ();
    Quaternion rotatedPoint = *this * Quaternion (point) * thisInverse;
    return (Vector3<float> (rotatedPoint.x, rotatedPoint.y, rotatedPoint.z));
  }

  // Operators
  Quaternion&
  Quaternion::operator+= (const Quaternion& q)
  {
    w += q.w;
    x += q.x;
    y += q.y;
    z += q.z;
    return (*this);
  }

  Quaternion&
  Quaternion::operator-= (const Quaternion& q)
  {
    w -= q.w;
    x -= q.x;
    y -= q.y;
    z -= q.z;
    return (*this);
  }

  Quaternion&
  Quaternion::operator*= (const Quaternion& q)
  {
    // (w1 + v1)(w2 + v2)
    //   = (w1 * w2 - v1 . v2, w1 * v2 + w2 * v1 + v1 x v2)
    float wr = w * q.w - (x * q.x + y * q.y + z * q.z);
    float xr = w * q.x + x * q.w + y * q.z - z * q.y;
    float yr = w * q.y + y * q.w - x * q.z + z * q.x;
    float zr = w * q.z + z * q.w + x * q.y - y * q.x;
    w = wr;
    x = xr;
    y = yr;
    z = zr;
    return (*this);
  }

  Quaternion&
  Quaternion::operator*= (float scalar)
  {
    w *= scalar;
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return (*this);
  }

  Quaternion&
  Quaternion::operator/= (float scalar)
  {
    float invScalar = 1 / scalar;
    w *= invScalar;
    x *= invScalar;
    y *= invScalar;
    z *= invScalar;
    return (*this);
  }
}
