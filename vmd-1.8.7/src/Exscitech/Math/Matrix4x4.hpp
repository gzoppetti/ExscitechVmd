#ifndef MATRIX4X4_HPP_
#define MATRIX4X4_HPP_

#include <algorithm>
#include <iostream>
#include <cstring>

#include "Exscitech/Types.hpp"

#include "Exscitech/Math/Matrix3x4.hpp"
#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector4.hpp"

namespace Exscitech
{
  template<typename T>
    class Matrix4x4
    {
    public:

      Matrix4x4 ()
      {
      }

      explicit
      Matrix4x4 (bool makeIdentity) :
          m_00 (1), m_01 (0), m_02 (0), m_03 (0), m_10 (0), m_11 (1), m_12 (0), m_13 (
              0), m_20 (0), m_21 (0), m_22 (1), m_23 (0), m_30 (0), m_31 (0), m_32 (
              0), m_33 (1)
      {
        // Don't care about the value of "makeIdentity"
      }

      explicit
      Matrix4x4 (const T array[16]) :
          m_00 (array[0]), m_01 (array[1]), m_02 (array[2]), m_03 (array[3]), m_10 (
              array[4]), m_11 (array[5]), m_12 (array[6]), m_13 (array[7]), m_20 (
              array[8]), m_21 (array[9]), m_22 (array[10]), m_23 (array[11]), m_30 (
              array[12]), m_31 (array[13]), m_32 (array[14]), m_33 (array[15])
      {
      }

      explicit
      Matrix4x4 (const Matrix3x4<T>& matrix3x4)
      {
        matrix3x4.getAsArray (&m_matrix[0][0]);
      }

      explicit
      Matrix4x4 (const Vector3<T>& right, const Vector3<T>& up,
          const Vector3<T>& back, const Vector3<T>& translation) :
          m_00 (right.x), m_01 (right.y), m_02 (right.z), m_03 (0), m_10 (up.x), m_11 (
              up.y), m_12 (up.z), m_13 (0), m_20 (back.x), m_21 (back.y), m_22 (
              back.z), m_23 (0), m_30 (translation.x), m_31 (translation.y), m_32 (
              translation.z), m_33 (1)
      {
      }

      void
      setToIdentity ()
      {
        setToZero ();
        m_00 = 1;
        m_11 = 1;
        m_22 = 1;
        m_33 = 1;
      }

      void
      setToZero ()
      {
        memset (&m_matrix[0][0], 0, 16 * sizeof(T));
      }

      void
      getAsArray (T array[16]) const
      {
        memcpy (array, &m_matrix[0][0], 16 * sizeof(T));
      }

      void
      setRight (const Vector3<T>& right)
      {
        m_00 = right.x;
        m_01 = right.y;
        m_02 = right.z;
        m_03 = 0;
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
        m_13 = 0;
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
        m_23 = 0;
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
        m_23 = 0;
      }

      Vector3<T>
      getForward () const
      {
        return (Vector3<T> (-m_20, -m_21, -m_22));
      }

      void
      setTranslation (const Vector3<T>& trans)
      {
        m_30 = trans.x;
        m_31 = trans.y;
        m_32 = trans.z;
        // It's a point
        m_33 = 1;
      }

      Vector3<T>
      getTranslation () const
      {
        return (Vector3<T> (m_30, m_31, m_32));
      }

      Matrix3x4<T>
      getTransform () const
      {
        Matrix3x4<T> transform (&m_matrix[0][0]);
        return (transform);
      }

      void
      transpose ()
      {
        std::swap (m_10, m_01);
        std::swap (m_20, m_02);
        std::swap (m_21, m_12);
        std::swap (m_30, m_03);
        std::swap (m_31, m_13);
        std::swap (m_32, m_23);
      }

      void
      negate ()
      {
        m_00 = -m_00;
        m_01 = -m_01;
        m_02 = -m_02;
        m_03 = -m_03;
        m_10 = -m_10;
        m_11 = -m_11;
        m_12 = -m_12;
        m_13 = -m_13;
        m_20 = -m_20;
        m_21 = -m_21;
        m_22 = -m_23;
        m_23 = -m_23;
        m_30 = -m_30;
        m_31 = -m_31;
        m_32 = -m_33;
        m_33 = -m_33;
      }

      void
      setToPerspectiveProjection (T fovYDegrees, T aspectRatio, T zNear, T zFar)
      {
        T fovYOver2Rads = Math<T>::toRadians (fovYDegrees / 2);
        T f = 1 / tan (fovYOver2Rads);
        m_00 = f / aspectRatio;
        m_01 = m_02 = m_03 = 0;
        m_10 = 0;
        m_11 = f;
        m_12 = m_13 = 0;
        m_20 = m_21 = 0;
        // Clip space is left-handed
        m_22 = (zFar + zNear) / (zNear - zFar);
        m_23 = -1;
        m_30 = m_31 = 0;
        m_32 = 2 * zFar * zNear / (zNear - zFar);
        m_33 = 0;
      }

      Vector4<T>
      transform (const Vector4<T>& point) const
      {
        Vector4<T> prod (
            m_00 * point.x + m_10 * point.y + m_20 * point.z + m_30,
            m_01 * point.x + m_11 * point.y + m_21 * point.z + m_31,
            m_02 * point.x + m_12 * point.y + m_22 * point.z + m_32,
            m_03 * point.x + m_13 * point.y + m_23 * point.z + m_33);
        return (prod);
      }

      // Operators
      T&
      operator() (uchar i, uchar j)
      {
        return (m_matrix[i][j]);
      }

      const T&
      operator() (uchar i, uchar j) const
      {
        return (m_matrix[i][j]);
      }

      Matrix4x4&
      operator+= (const Matrix4x4& m)
      {
        m_00 += m.m_00;
        m_01 += m.m_01;
        m_02 += m.m_02;
        m_03 += m.m_03;
        m_10 += m.m_10;
        m_11 += m.m_11;
        m_12 += m.m_12;
        m_13 += m.m_13;
        m_20 += m.m_20;
        m_21 += m.m_21;
        m_22 += m.m_22;
        m_23 += m.m_23;
        m_30 += m.m_30;
        m_31 += m.m_31;
        m_32 += m.m_32;
        m_33 += m.m_33;
        return (*this);
      }

      Matrix4x4&
      operator-= (const Matrix4x4& m)
      {
        m_00 -= m.m_00;
        m_01 -= m.m_01;
        m_02 -= m.m_02;
        m_03 -= m.m_03;
        m_10 -= m.m_10;
        m_11 -= m.m_11;
        m_12 -= m.m_12;
        m_13 -= m.m_13;
        m_20 -= m.m_20;
        m_21 -= m.m_21;
        m_22 -= m.m_22;
        m_23 -= m.m_23;
        m_30 -= m.m_30;
        m_31 -= m.m_31;
        m_32 -= m.m_32;
        m_33 -= m.m_33;
        return (*this);
      }

      Matrix4x4&
      operator*= (const Matrix4x4& m)
      {
        // Because matrices are transposed:
        //   (product_ij)' = dot (col_i (this), row_j (m))
        T r00 = m_00 * m.m_00 + m_10 * m.m_01 + m_20 * m.m_02 + m_30 * m.m_03;
        T r01 = m_00 * m.m_10 + m_10 * m.m_11 + m_20 * m.m_12 + m_30 * m.m_13;
        T r02 = m_00 * m.m_20 + m_10 * m.m_21 + m_20 * m.m_22 + m_30 * m.m_23;
        T r03 = m_00 * m.m_30 + m_10 * m.m_31 + m_20 * m.m_32 + m_30 * m.m_33;
        T r10 = m_01 * m.m_00 + m_11 * m.m_01 + m_21 * m.m_02 + m_31 * m.m_03;
        T r11 = m_01 * m.m_10 + m_11 * m.m_11 + m_21 * m.m_12 + m_31 * m.m_13;
        T r12 = m_01 * m.m_20 + m_11 * m.m_21 + m_21 * m.m_22 + m_31 * m.m_23;
        T r13 = m_01 * m.m_30 + m_11 * m.m_31 + m_21 * m.m_32 + m_31 * m.m_33;
        T r20 = m_02 * m.m_00 + m_12 * m.m_01 + m_22 * m.m_02 + m_32 * m.m_03;
        T r21 = m_02 * m.m_10 + m_12 * m.m_11 + m_22 * m.m_12 + m_32 * m.m_13;
        T r22 = m_02 * m.m_20 + m_12 * m.m_21 + m_22 * m.m_22 + m_32 * m.m_23;
        T r23 = m_02 * m.m_30 + m_12 * m.m_31 + m_22 * m.m_32 + m_32 * m.m_33;
        T r30 = m_03 * m.m_00 + m_13 * m.m_01 + m_23 * m.m_02 + m_33 * m.m_03;
        T r31 = m_03 * m.m_10 + m_13 * m.m_11 + m_23 * m.m_12 + m_33 * m.m_13;
        T r32 = m_03 * m.m_20 + m_13 * m.m_21 + m_23 * m.m_22 + m_33 * m.m_23;
        T r33 = m_03 * m.m_30 + m_13 * m.m_31 + m_23 * m.m_32 + m_33 * m.m_33;
        m_00 = r00;
        m_01 = r10;
        m_02 = r20;
        m_03 = r30;
        m_10 = r01;
        m_11 = r11;
        m_12 = r21;
        m_13 = r31;
        m_20 = r02;
        m_21 = r12;
        m_22 = r22;
        m_23 = r32;
        m_30 = r03;
        m_31 = r13;
        m_32 = r23;
        m_33 = r33;
        return (*this);
      }

    private:

      union
      {
        T m_matrix[4][4];
        struct
        {
          T m_00, m_01, m_02, m_03;
          T m_10, m_11, m_12, m_13;
          T m_20, m_21, m_22, m_23;
          T m_30, m_31, m_32, m_33;
        };
      };

    };

  template<typename T>
    inline Matrix4x4<T>
    operator+ (const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
      Matrix4x4<T> sum (m1);
      sum += m2;
      return (sum);
    }

  template<typename T>
    inline Matrix4x4<T>
    operator- (const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
      Matrix4x4<T> diff (m1);
      diff -= m2;
      return (diff);
    }

  // Unary negation
  template<typename T>
    inline Matrix4x4<T>
    operator- (const Matrix4x4<T>& m)
    {
      Matrix4x4<T> negated (m);
      negated.negate ();
      return (negated);
    }

  template<typename T>
    inline Matrix4x4<T>
    operator* (const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
      Matrix4x4<T> prod (m1);
      prod *= m2;
      return (prod);
    }

  template<typename T>
    std::ostream&
    operator<< (std::ostream& outStream, const Matrix4x4<T>& m)
    {
      std::ios_base::fmtflags origState = outStream.flags ();
      outStream << std::right << std::setprecision (2) << std::showpoint;
      outStream << std::setw (10) << m (0, 0) << std::setw (10) << m (0, 1)
          << std::setw (10) << m (0, 2) << std::setw (10) << m (0, 3) << '\n';
      outStream << std::setw (10) << m (1, 0) << std::setw (10) << m (1, 1)
          << std::setw (10) << m (1, 2) << std::setw (10) << m (1, 3) << '\n';
      outStream << std::setw (10) << m (2, 0) << std::setw (10) << m (2, 1)
          << std::setw (10) << m (2, 2) << std::setw (10) << m (2, 3) << '\n';
      outStream << std::setw (10) << m (3, 0) << std::setw (10) << m (3, 1)
          << std::setw (10) << m (3, 2) << std::setw (10) << m (3, 3) << '\n';
      outStream.flags (origState);
      return (outStream);
    }

  typedef Matrix4x4<float> Matrix4x4f;
  typedef Matrix4x4<Double> Matrix4x4d;

}
#endif
