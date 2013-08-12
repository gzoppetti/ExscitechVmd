#ifndef VECTOR4_HPP_
#define VECTOR4_HPP_

#include <cmath>
#include <iostream>
#include <iomanip>

#include "Exscitech/Math/Math.hpp"

namespace Exscitech
{
  template<typename T>
    class Vector4
    {
    public:

      Vector4 () :
        x (0), y (0), z (0), w (0)
      {
      }

      Vector4 (T xyzw) :
        x (xyzw), y (xyzw), z (xyzw), w (xyzw)
      {
      }

      Vector4 (T px, T py, T pz, T pw) :
        x (px), y (py), z (pz), w (pw)
      {
      }

      Vector4 (const T* const arrayOf4) :
        x (arrayOf4[0]), y (arrayOf4[1]), z (arrayOf4[2]), w (arrayOf4[3])
      {
      }

      Vector4 (const Vector3<T>& vec3, T pw) :
        x (vec3.x), y (vec3.y), z (vec3.z), w (pw)
      {
      }

      void
      set (T xyzw)
      {
        this->x = this->y = this->z = this->w = xyzw;
      }

      void
      set (T x, T y, T z, T w)
      {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
      }

      void
      set (const Vector3<T>& vec3, T w)
      {
        this->x = vec3.x;
        this->y = vec3.y;
        this->z = vec3.z;
        this->w = w;
      }

      void
      setToZero ()
      {
        x = y = z = w = 0;
      }

      void
      setToOne ()
      {
        x = y = z = w = 1;
      }

      T
      dot (const Vector4& v) const
      {
        return (x * v.x + y * v.y + z * v.z + w * v.w);
      }

      T
      lengthSquared () const
      {
        return (dot (*this));
      }

      T
      length () const
      {
        return (static_cast<T> (sqrt (lengthSquared ())));
      }

      T
      normalize ()
      {
        T lLength = length ();
        T invLength = 1 / lLength;
        x *= invLength;
        y *= invLength;
        z *= invLength;
        w *= invLength;
        return (lLength);
      }

      Vector4
      project (const Vector4& v, bool isVUnit = true) const
      {
        T mag = dot (v);
        if (!isVUnit)
        {
          mag *= 1 / v.lengthSquared ();
        }
        return (mag * v);
      }

      Vector4
      lerp (const Vector4& target, T targetWeight) const
      {
        return (Vector4 (x + (target.x - x) * targetWeight,
            y + (target.y - y) * targetWeight,
            z + (target.z - z) * targetWeight,
            w + (target.w - w) * targetWeight));
      }

      Vector4&
      multiplyComponents (const Vector4& v)
      {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return (*this);
      }

      // Operators

      T&
      operator[] (size_t i)
      {
        return (coords[i]);
      }

      const T&
      operator[] (size_t i) const
      {
        return (coords[i]);
      }

      Vector4&
      operator+= (const Vector4& v)
      {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return (*this);
      }

      Vector4&
      operator-= (const Vector4& v)
      {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return (*this);
      }

      Vector4&
      operator*= (T scalar)
      {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return (*this);
      }

      Vector4&
      operator/= (T scalar)
      {
        T invScalar = 1 / scalar;
        x *= invScalar;
        y *= invScalar;
        z *= invScalar;
        w *= invScalar;
        return (*this);
      }

    public:

      union
      {
        T coords[4];
        struct
        {
          T x, y, z, w;
        };
      };

    };

  template<typename T>
    inline
    bool
    operator== (const Vector4<T>& v1, const Vector4<T>& v2)
    {
      return (Math<T>::equalsUsingTolerance (v1.x, v2.x)
          && Math<T>::equalsUsingTolerance (v1.y, v2.y)
          && Math<T>::equalsUsingTolerance (v1.z, v2.z)
          && Math<T>::equalsUsingTolerance (v1.w, v2.w));
    }

  template<typename T>
    inline
    bool
    operator!= (const Vector4<T>& v1, const Vector4<T>& v2)
    {
      return (!(v1 == v2));
    }

  template<typename T>
    inline Vector4<T>
    operator+ (const Vector4<T>& v1, const Vector4<T>& v2)
    {
      Vector4<T> sum (v1);
      sum += v2;
      return (sum);
    }

  template<typename T>
    inline Vector4<T>
    operator- (const Vector4<T>& v1, const Vector4<T>& v2)
    {
      Vector4<T> diff (v1);
      diff -= v2;
      return (diff);
    }

  template<typename T>
    inline Vector4<T>
    operator- (const Vector4<T>& v)
    {
      return (Vector4<T> (-v.x, -v.y, -v.z, -v.w));
    }

  template<typename T>
    inline Vector4<T>
    multiplyComponents (const Vector4<T>& v1, const Vector4<T>& v2)
    {
      Vector4<T> prod (v1);
      prod *= v2;
      return (prod);
    }

  template<typename T>
    inline Vector4<T>
    operator* (const Vector4<T>& v, T scalar)
    {
      Vector4<T> multiple (v);
      multiple *= scalar;
      return (multiple);
    }

  template<typename T>
    inline Vector4<T>
    operator* (T scalar, const Vector4<T>& v)
    {
      return (v * scalar);
    }

  template<typename T>
    inline Vector4<T>
    operator/ (const Vector4<T>& v, T scalar)
    {
      Vector4<T> quotient (v);
      quotient /= scalar;
      return (quotient);
    }

  template<typename T>
    std::ostream&
    operator<< (std::ostream& outStream, const Vector4<T>& v)
    {
      std::ios_base::fmtflags origState = outStream.flags ();
      outStream << std::setprecision (2) << std::showpoint;
      outStream << "[ " << v.x << ", " << v.y << ", " << v.z << ", " << v.w
          << " ]";
      outStream.flags (origState);
      return (outStream);
    }

  typedef Vector4<float> Vector4f;
  typedef Vector4<double> Vector4d;
  typedef Vector4<int> Vector4i;
  typedef Vector4<unsigned int> Vector4u;
}

#endif

