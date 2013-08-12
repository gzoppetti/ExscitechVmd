#ifndef VECTOR3_HPP_
#define VECTOR3_HPP_

#include <cmath>
#include <iostream>
#include <iomanip>

#include <bullet/LinearMath/btVector3.h>

#include "Exscitech/Math/Math.hpp"

namespace Exscitech
{
  template<typename T>
    class Vector3
    {
    public:

      Vector3 () :
          x (0), y (0), z (0)
      {
      }

      explicit
      Vector3 (T xyz) :
          x (xyz), y (xyz), z (xyz)
      {
      }

      Vector3 (T px, T py, T pz) :
          x (px), y (py), z (pz)
      {
      }

      explicit
      Vector3 (const T arrayOf3[3]) :
          x (arrayOf3[0]), y (arrayOf3[1]), z (arrayOf3[2])
      {
      }

      void
      getAsArray (T arrayOf3[3]) const
      {
        arrayOf3[0] = x;
        arrayOf3[1] = y;
        arrayOf3[2] = z;
      }

      btVector3
      toBtVector3 () const
      {
        return (btVector3 (x, y, z));
      }

      void
      set (T xyz)
      {
        this->x = this->y = this->z = xyz;
      }

      void
      set (T x, T y, T z)
      {
        this->x = x;
        this->y = y;
        this->z = z;
      }

      void
      set (const Vector3<T>& source)
      {
        x = source.x;
        y = source.y;
        z = source.z;
      }

      void
      setToZero ()
      {
        x = y = z = 0;
      }

      void
      setToOne ()
      {
        x = y = z = 1;
      }

      void
      setToPerpendicularVector (const Vector3& v)
      {
        *this = v;
        if (x != 0)
        {
          std::swap (x, y);
          x = -x;
          z = 0;
        }
        else
        {
          std::swap (y, z);
          y = -y;
        }
      }

      T
      dot (const Vector3& v) const
      {
        return (x * v.x + y * v.y + z * v.z);
      }

      Vector3
      cross (const Vector3& v) const
      {
        T rx = y * v.z - z * v.y;
        T ry = -(x * v.z - z * v.x);
        T rz = x * v.y - y * v.x;
        return (Vector3 (rx, ry, rz));
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
        return (lLength);
      }

      Vector3
      project (const Vector3& v, bool isVUnit = true) const
      {
        T mag = dot (v);
        if (!isVUnit)
        {
          mag *= 1 / v.lengthSquared ();
        }
        return (mag * v);
      }

      Vector3
      lerp (const Vector3& target, T targetWeight) const
      {
        return (Vector3 (x + (target.x - x) * targetWeight,
            y + (target.y - y) * targetWeight,
            z + (target.z - z) * targetWeight));
      }

      Vector3&
      multiplyComponents (const Vector3& v)
      {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return (*this);
      }

      void
      negate ()
      {
        x = -x;
        y = -y;
        z = -z;
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

      Vector3&
      operator+= (const Vector3& v)
      {
        x += v.x;
        y += v.y;
        z += v.z;
        return (*this);
      }

      Vector3&
      operator-= (const Vector3& v)
      {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return (*this);
      }

      Vector3&
      operator*= (T scalar)
      {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return (*this);
      }

      Vector3&
      operator/= (T scalar)
      {
        T invScalar = 1 / scalar;
        x *= invScalar;
        y *= invScalar;
        z *= invScalar;
        return (*this);
      }

    public:

      union
      {
        T coords[3];
        struct
        {
          T x, y, z;
        };
      };

    };

  template<typename T>
    inline float
    computeDistanceBetween (const Vector3<T>& v1, const Vector3<T>& v2)
    {
      Vector3<T> delta = v1 - v2;
      return (delta.length ());
    }

  template<typename T>
    inline float
    computeDistanceSquaredBetween (const Vector3<T>& v1, const Vector3<T>& v2)
    {
      Vector3<T> delta = v1 - v2;
      return (delta.lengthSquared ());
    }

  template<typename T>
    inline float
    computeAngleBetween (const Vector3<T>& v1, const Vector3<T>& v2,
        bool unitVectors)
    {
      float cosTheta = v1.dot (v2);
      if (!unitVectors)
      {
        cosTheta /= v1.length () * v2.length ();
      }
      float theta = acos (cosTheta);
      return (theta);
    }

  template<typename T>
    inline float
    computeMaxComponent (const Vector3<T>& v1, T epsilon)
    {
      T max1 = Math<T>::maxUsingTolerance (v1.x, v1.y, epsilon);
      T max2 = Math<T>::maxUsingTolerance (max1, v1.z, epsilon);
      return max2;
    }

  template<typename T>
    inline Vector3<T>
    clamp (const Vector3<T>& v, const Vector3<T>& lowerBound,
        const Vector3<T>& upperBound)
    {
      return Vector3<T> (Math<T>::clamp (v.x, lowerBound.x, upperBound.x),
          Math<T>::clamp (v.y, lowerBound.y, upperBound.y),
          Math<T>::clamp (v.z, lowerBound.z, upperBound.z));
    }
  template<typename T>
    inline
    bool
    operator== (const Vector3<T>& v1, const Vector3<T>& v2)
    {
      return (Math<T>::equalsUsingTolerance (v1.x, v2.x)
          && Math<T>::equalsUsingTolerance (v1.y, v2.y)
          && Math<T>::equalsUsingTolerance (v1.z, v2.z));
    }

  template<typename T>
    inline
    bool
    operator!= (const Vector3<T>&v1, const Vector3<T>& v2)
    {
      return (!(v1 == v2));
    }

  template<typename T>
    inline Vector3<T>
    operator+ (const Vector3<T>& v1, const Vector3<T>& v2)
    {
      Vector3<T> sum (v1);
      sum += v2;
      return (sum);
    }

  template<typename T>
    inline Vector3<T>
    operator- (const Vector3<T>& v1, const Vector3<T>& v2)
    {
      Vector3<T> diff (v1);
      diff -= v2;
      return (diff);
    }

  template<typename T>
    inline Vector3<T>
    operator- (const Vector3<T>& v)
    {
      Vector3<T> negated (v);
      negated.negate ();
      return (negated);
    }

  template<typename T>
    inline Vector3<T>
    multiplyComponents (const Vector3<T>& v1, const Vector3<T>& v2)
    {
      Vector3<T> prod (v1);
      prod *= v2;
      return (prod);
    }

  template<typename T>
    inline Vector3<T>
    operator* (const Vector3<T>& v, T scalar)
    {
      Vector3<T> multiple (v);
      multiple *= scalar;
      return (multiple);
    }

  template<typename T>

    Vector3<T>
    operator* (T scalar, const Vector3<T>& v)
    {
      return (v * scalar);
    }

  template<typename T>
    inline Vector3<T>
    operator/ (const Vector3<T>& v, T scalar)
    {
      Vector3<T> quotient (v);
      quotient /= scalar;
      return (quotient);
    }

  template<typename T>
    inline std::ostream&
    operator<< (std::ostream& outStream, const Vector3<T>& v)
    {
      std::ios_base::fmtflags origState = outStream.flags ();
      outStream << std::setprecision (2) << std::showpoint;
      outStream << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
      outStream.flags (origState);
      return (outStream);
    }

  typedef Vector3<float> Vector3f;
  typedef Vector3<double> Vector3d;
  typedef Vector3<int> Vector3i;
  typedef Vector3<unsigned int> Vector3u;
}

#endif

