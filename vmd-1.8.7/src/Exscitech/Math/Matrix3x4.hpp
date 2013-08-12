#ifndef MATRIX3X4_HPP_
#define MATRIX3X4_HPP_

#include "Exscitech/Math/Matrix3x3.hpp"
#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
  template<typename T>
    class Matrix3x4
    {
    public:

      Matrix3x4 ()
      {
      }

      Matrix3x4 (bool makeIdentity) :
          rotation (true), translation (static_cast<T> (0))
      {
        // Don't care about the value of "makeIdentity"
      }

      Matrix3x4 (const Matrix3x3<T>& pRotation, const Vector3<T>& pTranslation) :
          rotation (pRotation), translation (pTranslation)
      {
      }

      Matrix3x4 (T e00, T e01, T e02, T e03, T e10, T e11, T e12, T e13, T e20,
          T e21, T e22, T e23) :
          rotation (e00, e01, e02, e10, e11, e12, e20, e21, e22), translation (
              e03, e13, e23)
      {
      }

      Matrix3x4 (const T* const arrayOf16) :
          rotation (arrayOf16), translation (arrayOf16[3], arrayOf16[7],
              arrayOf16[11])
      {
      }

      Matrix3x4 (const Vector3<T>& right, const Vector3<T>& up,
          const Vector3<T>& backward) :
          rotation (right, up, backward), translation (static_cast<T> (0))
      {
      }

      Matrix3x4 (const Vector3<T>& right, const Vector3<T>& up,
          const Vector3<T>& backward, const Vector3<T>& translation) :
          rotation (right, up, backward), translation (translation)
      {
      }

      Matrix3x4 (const Vector3<T>& right, const Vector3<T>& up,
          const Vector3<T>& translation, bool ensureOrthonormal) :
          rotation (right, up, ensureOrthonormal), translation (translation)
      {
      }

      void
      setToIdentity ()
      {
        rotation.setToIdentity ();
        translation.setToZero ();
      }

      void
      setToZero ()
      {
        rotation.setToZero ();
        translation.setToZero ();
      }

      void
      scaleLocal (T scale)
      {
        rotation.scaleLocal (scale);
      }

      void
      scaleLocal (T scaleX, T scaleY, T scaleZ)
      {
        rotation.scaleLocal (scaleX, scaleY, scaleZ);
      }

      void
      scaleWorld (T scale)
      {
        rotation *= scale;
        translation *= scale;
      }

      void
      scaleWorld (T scaleX, T scaleY, T scaleZ)
      {
        rotation.scaleWorld (scaleX, scaleY, scaleZ);
        translation.x *= scaleX;
        translation.y *= scaleY;
        translation.z *= scaleZ;
      }

      // If rotation component is a pure rotation matrix,
      //   use this method for speed
      void
      invertRotationMatrix ()
      {
        rotation.invertRotationMatrix ();
        translation = rotation * -translation;
      }

      void
      invert ()
      {
        // [ M t ] [ M' M'(-t) ] = [ I 0 ]
        rotation.invert ();
        translation = rotation * -translation;
      }

      void
      getAsArray (T* const arrayOf16) const
      {
        rotation.getAsArray (arrayOf16);
        arrayOf16[12] = translation.x;
        arrayOf16[13] = translation.y;
        arrayOf16[14] = translation.z;
        arrayOf16[15] = 1;
      }

      void
      negate ()
      {
        rotation.negate ();
        translation.negate ();
      }

      Vector3<T>
      transform (const Vector3<T>& point) const
      {
        Vector3<T> prod = rotation.transform (point) + translation;
        return (prod);
      }

      Vector3<T>
      transformVector (const Vector3<T>& vector) const
      {
        // For a vector the translation is ignored
        Vector3<T> prod = rotation.transform (vector);
        return (prod);
      }

      // Operators
      Matrix3x4&
      operator+= (const Matrix3x4& m)
      {
        rotation += m.rotation;
        translation += m.translation;
        return (*this);
      }

      Matrix3x4&
      operator-= (const Matrix3x4& m)
      {
        rotation -= m.rotation;
        translation -= m.translation;
        return (*this);
      }

      Matrix3x4&
      operator*= (const Matrix3x4& m)
      {
        // [ M1 t1 ] [ M2 t2 ] = [ M1*M2 M1*t2+t1 ]
        // Order matters! Do not reverse following statements
        translation += rotation * m.translation;
        rotation *= m.rotation;
        return (*this);
      }

    public:

      Matrix3x3<T> rotation;
      Vector3<T> translation;

    };

  template<typename T>
    inline Matrix3x4<T>
    operator+ (const Matrix3x4<T>& m1, const Matrix3x4<T>& m2)
    {
      Matrix3x4<T> sum (m1);
      sum += m2;
      return (sum);
    }

  template<typename T>
    inline Matrix3x4<T>
    operator- (const Matrix3x4<T>& m1, const Matrix3x4<T>& m2)
    {
      Matrix3x4<T> diff (m1);
      diff -= m2;
      return (diff);
    }

  // Unary negation
  template<typename T>
    inline Matrix3x4<T>
    operator- (const Matrix3x4<T>& m)
    {
      Matrix3x4<T> negated (m);
      negated.negate ();
      return (negated);
    }

  template<typename T>
    inline Matrix3x4<T>
    operator* (const Matrix3x4<T>& m1, const Matrix3x4<T>& m2)
    {
      Matrix3x4<T> prod (m1);
      prod *= m2;
      return (prod);
    }

  template<typename T>
    inline Vector3<T>
    operator* (const Matrix3x4<T>& m, const Vector3<T>& point)
    {
      return (m.transform (point));
    }

  template<typename T>
    std::ostream&
    operator<< (std::ostream& outStream, const Matrix3x4<T>& m)
    {
      outStream << m.rotation;
      outStream << m.translation << '\n';
      return (outStream);
    }

  typedef Matrix3x4<float> Matrix3x4f;
  typedef Matrix3x4<double> Matrix3x4d;

}
#endif
