#ifndef MATRIX3X3_HPP_
#define MATRIX3X3_HPP_

#include <iostream>
#include <iomanip>
#include <cstdio>

#include "Exscitech/Types.hpp"

#include "Exscitech/Math/Math.hpp"
#include "Exscitech/Math/Quaternion.hpp"
#include "Exscitech/Math/Vector3.hpp"


namespace Exscitech
{
  // Basis vectors are stored in rows for efficiency
  // Operations are consistent with OpenGL (M * v = v')
  template<typename T>
    class Matrix3x3
    {
    public:

      Matrix3x3 ()
      {
      }

      explicit
      Matrix3x3 (bool makeIdentity) :
        m_00 (1), m_01 (0), m_02 (0), m_10 (0), m_11 (1), m_12 (0), m_20 (0),
            m_21 (0), m_22 (1)
      {
        // Don't care about the value of "makeIdentity"
      }

      Matrix3x3 (T e00, T e01, T e02, T e10, T e11, T e12, T e20, T e21, T e22) :
        m_00 (e00), m_01 (e01), m_02 (e02), m_10 (e10), m_11 (e11), m_12 (e12),
            m_20 (e20), m_21 (e21), m_22 (e22)
      {
      }

      Matrix3x3 (const T arrayOf16[16]) :
        m_00 (arrayOf16[0]), m_01 (arrayOf16[1]), m_02 (arrayOf16[2]),
            m_10 (arrayOf16[4]), m_11 (arrayOf16[5]), m_12 (arrayOf16[6]),
            m_20 (arrayOf16[8]), m_21 (arrayOf16[9]), m_22 (arrayOf16[10])
      {
      }

      Matrix3x3 (const Vector3<T>& right, const Vector3<T>& up,
          const Vector3<T>& backward) :
        m_00 (right.x), m_01 (right.y), m_02 (right.z), m_10 (up.x),
            m_11 (up.y), m_12 (up.z), m_20 (backward.x), m_21 (backward.y),
            m_22 (backward.z)
      {
      }

      Matrix3x3 (const Vector3<T>& right, const Vector3<T>& up,
          bool ensureOrthonormal = false) :
        m_00 (right.x), m_01 (right.y), m_02 (right.z), m_10 (up.x),
            m_11 (up.y), m_12 (up.z), m_20 (+right.y * up.z - right.z * up.y),
            m_21 (-right.x * up.z + right.z * up.x),
            m_22 (+right.x * up.y - right.y * up.x)
      {
        if (ensureOrthonormal)
        {
          orthonormalize ();
        }
      }

      void
      setToIdentity ()
      {
        m_00 = 1;
        m_01 = m_02 = 0;
        m_10 = 0;
        m_11 = 1;
        m_12 = 0;
        m_20 = m_21 = 0;
        m_22 = 1;
      }

      void
      setToZero ()
      {
        m_00 = m_01 = m_02 = 0;
        m_10 = m_11 = m_12 = 0;
        m_20 = m_21 = m_22 = 0;
      }

      void
      setFromMatrix (const float* const matrix4x4)
      {
        m_00 = *(matrix4x4 + 0);
        m_01 = *(matrix4x4 + 1);
        m_02 = *(matrix4x4 + 2);
        m_10 = *(matrix4x4 + 4);
        m_11 = *(matrix4x4 + 5);
        m_12 = *(matrix4x4 + 6);
        m_20 = *(matrix4x4 + 8);
        m_21 = *(matrix4x4 + 9);
        m_22 = *(matrix4x4 + 10);
      }

      Quaternion
      toQuaternion () const
      {
        Quaternion q;
        // Examine combinations of matrix's trace elements
        //   where trace = m00 + m11 + m22
        // Each expression "traceL" computes 4L^2 - 1
        float traceW = m_00 + m_11 + m_22;
        float traceX = m_00 - m_11 - m_22;
        float traceY = -m_00 + m_11 - m_22;
        float traceZ = -m_00 - m_11 + m_22;

        // Find largest magnitude component (w, x, y, or z)
        float traceLargest = traceW;
        uint largest = 0;
        if (traceX > traceLargest)
        {
          traceLargest = traceX;
          largest = 1;
        }
        if (traceY > traceLargest)
        {
          traceLargest = traceY;
          largest = 2;
        }
        if (traceZ > traceLargest)
        {
          traceLargest = traceZ;
          largest = 3;
        }

        float largestComponent = sqrt (traceLargest + 1) * 0.5f;
        // Factor = 1 / (4 * largestComponent)
        float factor = 0.25f / largestComponent;
        switch (largest)
        {
          case 0:
            q.w = largestComponent;
            q.x = (m_12 - m_21) * factor;
            q.y = (m_20 - m_02) * factor;
            q.z = (m_01 - m_10) * factor;
            break;
          case 1:
            q.x = largestComponent;
            q.w = (m_12 - m_21) * factor;
            q.y = (m_01 + m_10) * factor;
            q.z = (m_20 + m_02) * factor;
            break;
          case 2:
            q.y = largestComponent;
            q.w = (m_20 - m_02) * factor;
            q.x = (m_01 + m_10) * factor;
            q.z = (m_12 + m_21) * factor;
            break;
          case 3:
            q.z = largestComponent;
            q.w = (m_01 - m_10) * factor;
            q.x = (m_20 + m_02) * factor;
            q.y = (m_12 + m_21) * factor;
            break;
        }
        return (q);
      }

      void
      getAsArray (T array[16]) const
      {
        array[0] = m_00;
        array[1] = m_01;
        array[2] = m_02;
        array[3] = 0;
        array[4] = m_10;
        array[5] = m_11;
        array[6] = m_12;
        array[7] = 0;
        array[8] = m_20;
        array[9] = m_21;
        array[10] = m_22;
        array[11] = 0;
        array[12] = 0;
        array[13] = 0;
        array[14] = 0;
        array[15] = 1;
      }

      void
      setRight (const Vector3<T>& right)
      {
        m_00 = right.x;
        m_01 = right.y;
        m_02 = right.z;
      }

      Vector3<T>
      getRight () const
      {
        return (Vector3<T> (m_00, m_01, m_02));
      }

      void
      setUp (const Vector3<T>& up)
      {
        m_10 = up.x;
        m_11 = up.y;
        m_12 = up.z;
      }

      Vector3<T>
      getUp () const
      {
        return (Vector3<T> (m_10, m_11, m_12));
      }

      void
      setBackward (const Vector3<T>& backward)
      {
        m_20 = backward.x;
        m_21 = backward.y;
        m_22 = backward.z;
      }

      Vector3<T>
      getBackward () const
      {
        return (Vector3<T> (m_20, m_21, m_22));
      }

      void
      setForward (const Vector3<T>& forward)
      {
        m_20 = -forward.x;
        m_21 = -forward.y;
        m_22 = -forward.z;
      }

      Vector3<T>
      getForward () const
      {
        return (Vector3<T> (-m_20, -m_21, -m_22));
      }

      void
      invertRotationMatrix ()
      {
        this->transpose ();
      }

      void
      invert ()
      {
        T minor00, minor01, minor02;
        T minor10, minor11, minor12;
        T minor20, minor21, minor22;
        minor00 = m_11 * m_22 - m_12 * m_21;
        minor01 = m_10 * m_22 - m_12 * m_20;
        minor02 = m_10 * m_21 - m_11 * m_20;
        minor10 = m_01 * m_22 - m_02 * m_21;
        minor11 = m_00 * m_22 - m_02 * m_20;
        minor12 = m_00 * m_21 - m_01 * m_20;
        minor20 = m_01 * m_12 - m_02 * m_11;
        minor21 = m_00 * m_12 - m_02 * m_10;
        minor22 = m_00 * m_11 - m_01 * m_10;

        // Determinant could be zero for singular matrix
        T determinant = m_00 * minor00 - m_01 * minor01 + m_02 * minor02;
        T inverseDeterminant = 1 / determinant;

        m_00 = minor00 * inverseDeterminant;
        m_01 = -minor10 * inverseDeterminant;
        m_02 = minor20 * inverseDeterminant;
        m_10 = -minor01 * inverseDeterminant;
        m_11 = minor11 * inverseDeterminant;
        m_12 = -minor21 * inverseDeterminant;
        m_20 = minor02 * inverseDeterminant;
        m_21 = -minor12 * inverseDeterminant;
        m_22 = minor22 * inverseDeterminant;
      }

      void
      transpose ()
      {
        std::swap (m_01, m_10);
        std::swap (m_02, m_20);
        std::swap (m_12, m_21);
      }

      void
      orthonormalize ()
      {
        // Normalize backward
        Math<T>::normalize (&m_20);
        // right = up x backward
        Math<T>::cross (&m_00, &m_10, &m_20);
        // Normalize right
        Math<T>::normalize (&m_00);
        // up = backward x right
        Math<T>::cross (&m_10, &m_20, &m_00);
      }

      void
      extractYawPitchRoll (float& yawDegrees, float& pitchDegrees,
          float& rollDegrees)
      {
        if (m_12 < +1)
        {
          if (m_12 > -1)
          {
            pitchDegrees = Math<T>::toDegrees (asin (-m_21));
            yawDegrees = Math<T>::toDegrees (atan2 (m_20, m_22));
            rollDegrees = Math<T>::toDegrees (atan2 (m_01, m_11));
          }
          else
          {
            // m_12 = -1
            // Not a unique solution: roll - yaw = atan2 (-m_01, m_00)
            pitchDegrees = +90;
            yawDegrees = Math<T>::toDegrees (-atan2 (-m_10, m_00));
            rollDegrees = 0;
          }
        }
        else
        {
          // m_12 = +1
          // Not a unique solution: roll + yaw = atan2 (-m_01, m_00)
          pitchDegrees = -90;
          yawDegrees = Math<T>::toDegrees (atan2 (-m_10, m_00));
          rollDegrees = 0;
        }
      }

      void
      scaleLocal (T scale)
      {
        m_00 *= scale;
        m_01 *= scale;
        m_02 *= scale;
        m_10 *= scale;
        m_11 *= scale;
        m_12 *= scale;
        m_20 *= scale;
        m_21 *= scale;
        m_22 *= scale;
      }

      void
      scaleLocal (T scaleX, T scaleY, T scaleZ)
      {
        m_00 *= scaleX;
        m_01 *= scaleX;
        m_02 *= scaleX;
        m_10 *= scaleY;
        m_11 *= scaleY;
        m_12 *= scaleY;
        m_20 *= scaleZ;
        m_21 *= scaleZ;
        m_22 *= scaleZ;
      }

      void
      scaleWorld (T scale)
      {
        // No difference from scaleLocal w/o a translation
        scaleLocal (scale);
      }

      void
      scaleWorld (T scaleX, T scaleY, T scaleZ)
      {
        m_00 *= scaleX;
        m_01 *= scaleY;
        m_02 *= scaleZ;
        m_10 *= scaleX;
        m_11 *= scaleY;
        m_12 *= scaleZ;
        m_20 *= scaleX;
        m_21 *= scaleY;
        m_22 *= scaleZ;
      }

      void
      setToScale (T scale)
      {
        m_00 = m_11 = m_22 = scale;
        m_01 = m_02 = m_10 = m_12 = m_20 = m_21 = 0;
      }

      void
      setToScale (T scaleX, T scaleY, T scaleZ)
      {
        m_00 = scaleX;
        m_11 = scaleY;
        m_22 = scaleZ;
        m_01 = m_02 = m_10 = m_12 = m_20 = m_21 = 0;
      }

      void
      yaw (T angleDegrees)
      {
        Matrix3x3<T> rotation;
        rotation.setToRotationY (angleDegrees);
        *this *= rotation;
      }

      void
      pitch (T angleDegrees)
      {
        Matrix3x3<T> rotation;
        rotation.setToRotationX (angleDegrees);
        *this *= rotation;
      }

      void
      roll (T angleDegrees)
      {
        Matrix3x3<T> rotation;
        rotation.setToRotationZ (angleDegrees);
        *this *= rotation;
      }

      void
      rotateWorldY (T angleDegrees)
      {
        Matrix3x3<T> rotation;
        rotation.setToRotationY (angleDegrees);
        *this = rotation * *this;
      }

      void
      rotateWorldX (T angleDegrees)
      {
        Matrix3x3<T> rotation;
        rotation.setToRotationX (angleDegrees);
        *this = rotation * *this;
      }

      void
      rotateWorldZ (T angleDegrees)
      {
        Matrix3x3<T> rotation;
        rotation.setToRotationZ (angleDegrees);
        *this = rotation * *this;
      }

      void
      setToRotationY (T angleDegrees)
      {
        T angleRad = Math<T>::toRadians (angleDegrees);
        T cosAngle = cos (angleRad);
        T sinAngle = sin (angleRad);
        m_00 = cosAngle;
        m_01 = 0;
        m_02 = -sinAngle;
        m_10 = 0;
        m_11 = 1;
        m_12 = 0;
        m_20 = sinAngle;
        m_21 = 0;
        m_22 = cosAngle;
      }

      void
      setToRotationX (float angleDegrees)
      {
        float angleRad = Math<T>::toRadians (angleDegrees);
        float cosAngle = cos (angleRad);
        float sinAngle = sin (angleRad);
        m_00 = 1;
        m_01 = 0;
        m_02 = 0;
        m_10 = 0;
        m_11 = cosAngle;
        m_12 = sinAngle;
        m_20 = 0;
        m_21 = -sinAngle;
        m_22 = cosAngle;
      }

      void
      setToRotationZ (float angleDegrees)
      {
        float angleRad = Math<T>::toRadians (angleDegrees);
        float cosAngle = cos (angleRad);
        float sinAngle = sin (angleRad);
        m_00 = cosAngle;
        m_01 = sinAngle;
        m_02 = 0;
        m_10 = -sinAngle;
        m_11 = cosAngle;
        m_12 = 0;
        m_20 = 0;
        m_21 = 0;
        m_22 = 1;
      }

      void
      setFromYawPitchRoll (float yawDegrees, float pitchDegrees,
          float rollDegrees)
      {
        float yawRad = Math<T>::toRadians (yawDegrees);
        float pitchRad = Math<T>::toRadians (pitchDegrees);
        float rollRad = Math<T>::toRadians (rollDegrees);
        float cosYaw = cos (yawRad);
        float cosPitch = cos (pitchRad);
        float cosRoll = cos (rollRad);
        float sinYaw = sin (yawRad);
        float sinPitch = sin (pitchRad);
        float sinRoll = sin (rollRad);
        m_00 = cosYaw * cosRoll + sinPitch * sinYaw * sinRoll;
        m_01 = cosPitch * sinRoll;
        m_02 = -cosRoll * sinYaw + cosYaw * sinPitch * sinRoll;
        m_10 = cosRoll * sinPitch * sinYaw - cosYaw * sinRoll;
        m_11 = cosPitch * cosRoll;
        m_12 = cosYaw * cosRoll * sinPitch + sinYaw * sinRoll;
        m_20 = cosPitch * sinYaw;
        m_21 = -sinPitch;
        m_22 = cosPitch * cosYaw;
      }

      void
      setFromAngleAxis (float angleDegrees, const Vector3f& axis)
      {
        float angleRad = Math<T>::toRadians (angleDegrees);
        float cosV = cos (angleRad);
        float oneMinusCos = 1 - cosV;
        float sinV = sin (angleRad);

        m_00 = oneMinusCos * axis.x * axis.x + cosV;
        m_01 = oneMinusCos * axis.x * axis.y + axis.z * sinV;
        m_02 = oneMinusCos * axis.x * axis.z - axis.y * sinV;

        m_10 = oneMinusCos * axis.x * axis.y - axis.z * sinV;
        m_11 = oneMinusCos * axis.y * axis.y + cosV;
        m_12 = oneMinusCos * axis.y * axis.z + axis.x * sinV;

        m_20 = oneMinusCos * axis.x * axis.z + axis.y * sinV;
        m_21 = oneMinusCos * axis.y * axis.z - axis.x * sinV;
        m_22 = oneMinusCos * axis.z * axis.z + cosV;
      }

      void
      negate ()
      {
        m_00 = -m_00;
        m_01 = -m_01;
        m_02 = -m_02;
        m_10 = -m_10;
        m_11 = -m_11;
        m_12 = -m_12;
        m_20 = -m_20;
        m_21 = -m_21;
        m_22 = -m_22;
      }

      Vector3<T>
      transform (const Vector3<T>& point) const
      {
        Vector3<T> prod (m_00 * point.x + m_10 * point.y + m_20 * point.z,
            m_01 * point.x + m_11 * point.y + m_21 * point.z,
            m_02 * point.x + m_12 * point.y + m_22 * point.z);
        return (prod);
      }

      // Operators
      T&
      operator() (unsigned char i, unsigned char j)
      {
        return (m_matrix[i][j]);
      }

      const T&
      operator() (unsigned char i, unsigned char j) const
      {
        return (m_matrix[i][j]);
      }

      Matrix3x3&
      operator+= (const Matrix3x3& m)
      {
        m_00 += m.m_00;
        m_01 += m.m_01;
        m_02 += m.m_02;
        m_10 += m.m_10;
        m_11 += m.m_11;
        m_12 += m.m_12;
        m_20 += m.m_20;
        m_21 += m.m_21;
        m_22 += m.m_22;
        return (*this);
      }

      Matrix3x3&
      operator-= (const Matrix3x3& m)
      {
        m_00 -= m.m_00;
        m_01 -= m.m_01;
        m_02 -= m.m_02;
        m_10 -= m.m_10;
        m_11 -= m.m_11;
        m_12 -= m.m_12;
        m_20 -= m.m_20;
        m_21 -= m.m_21;
        m_22 -= m.m_22;
        return (*this);
      }

      Matrix3x3&
      operator*= (T scalar)
      {
        m_00 *= scalar;
        m_01 *= scalar;
        m_02 *= scalar;
        m_10 *= scalar;
        m_11 *= scalar;
        m_12 *= scalar;
        m_20 *= scalar;
        m_21 *= scalar;
        m_22 *= scalar;

        return (*this);
      }

      Matrix3x3&
      operator*= (const Matrix3x3& m)
      {
        // Because matrices are transposed:
        //   (product_ij)' = dot (col_i (this), row_j (m))
        // Let A = this' and B = m'
        // So we compute this = (A * B)'
        T r00 = m_00 * m.m_00 + m_10 * m.m_01 + m_20 * m.m_02;
        T r01 = m_00 * m.m_10 + m_10 * m.m_11 + m_20 * m.m_12;
        T r02 = m_00 * m.m_20 + m_10 * m.m_21 + m_20 * m.m_22;
        T r10 = m_01 * m.m_00 + m_11 * m.m_01 + m_21 * m.m_02;
        T r11 = m_01 * m.m_10 + m_11 * m.m_11 + m_21 * m.m_12;
        T r12 = m_01 * m.m_20 + m_11 * m.m_21 + m_21 * m.m_22;
        T r20 = m_02 * m.m_00 + m_12 * m.m_01 + m_22 * m.m_02;
        T r21 = m_02 * m.m_10 + m_12 * m.m_11 + m_22 * m.m_12;
        T r22 = m_02 * m.m_20 + m_12 * m.m_21 + m_22 * m.m_22;
        m_00 = r00;
        m_01 = r10;
        m_02 = r20;
        m_10 = r01;
        m_11 = r11;
        m_12 = r21;
        m_20 = r02;
        m_21 = r12;
        m_22 = r22;

        return (*this);
      }

    private:

      union
      {
        T m_matrix[3][3];
        struct
        {
          T m_00, m_01, m_02;
          T m_10, m_11, m_12;
          T m_20, m_21, m_22;
        };
      };

    };

  template<typename T>
    inline T
    dotWithMatrixColumn (const Vector3f& v, const Matrix3x3<T>& m,
        uchar columnNum)
    {
      return (v.x * m (0, columnNum) + v.y * m (1, columnNum) + v.z * m (2,
          columnNum));
    }

  template<typename T>
    inline Matrix3x3<T>
    operator+ (const Matrix3x3<T>& m1, const Matrix3x3<T>& m2)
    {
      Matrix3x3<T> sum (m1);
      sum += m2;
      return (sum);
    }

  template<typename T>
    inline Matrix3x3<T>
    operator- (const Matrix3x3<T>& m1, const Matrix3x3<T>& m2)
    {
      Matrix3x3<T> diff (m1);
      diff -= m2;
      return (diff);
    }

  // Unary negation
  template<typename T>
    inline Matrix3x3<T>
    operator- (const Matrix3x3<T>& m)
    {
      Matrix3x3<T> negated (m);
      negated.negate ();
      return (negated);
    }

  template<typename T>
    inline Matrix3x3<T>
    operator* (const Matrix3x3<T>& m, T scalar)
    {
      Matrix3x3<T> prod (m);
      prod *= scalar;
      return (prod);
    }

  template<typename T>
    inline Matrix3x3<T>
    operator* (T scalar, const Matrix3x3<T>& m)
    {
      return (m * scalar);
    }

  template<typename T>
    inline Matrix3x3<T>
    operator* (const Matrix3x3<T>& m1, const Matrix3x3<T>& m2)
    {
      Matrix3x3<T> prod (m1);
      prod *= m2;
      return (prod);
    }

  template<typename T>
    inline Vector3<T>
    operator* (const Matrix3x3<T>& m, const Vector3<T>& v)
    {
      return (m.transform (v));
    }

  template<typename T>
    std::ostream&
    operator<< (std::ostream& outStream, const Matrix3x3<T>& m)
    {
      std::ios_base::fmtflags origState = outStream.flags ();
      outStream << std::right << std::setprecision (2) << std::showpoint;
      outStream << std::setw (10) << m (0, 0) << std::setw (10) << m (0, 1)
          << std::setw (10) << m (0, 2) << '\n';
      outStream << std::setw (10) << m (1, 0) << std::setw (10) << m (1, 1)
          << std::setw (10) << m (1, 2) << '\n';
      outStream << std::setw (10) << m (2, 0) << std::setw (10) << m (2, 1)
          << std::setw (10) << m (2, 2) << '\n';
      outStream.flags (origState);
      return (outStream);
    }

  typedef Matrix3x3<float> Matrix3x3f;
  typedef Matrix3x3<double> Matrix3x3d;

}
#endif
