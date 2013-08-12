#ifndef VECTOR2_HPP_
#define VECTOR2_HPP_

#include <cmath>
#include <iostream>
#include <iomanip>

#include "Exscitech/Types.hpp"
#include "Exscitech/Math/Math.hpp"

namespace Exscitech
{
  template<typename T>
    class Vector2
    {
    public:

      Vector2 () :
        x (0), y (0)
      {
      }

      Vector2 (T xy) :
        x (xy), y (xy)
      {
      }

      Vector2 (T px, T py) :
        x (px), y (py)
      {
      }

      Vector2 (const T* const arrayOf2) :
        x (arrayOf2[0]), y (arrayOf2[1])
      {
      }

      void
      set (T xy)
      {
        this->x = this->y = xy;
      }

      void
      set (T x, T y)
      {
        this->x = x;
        this->y = y;
      }

      void
      setToZero ()
      {
        x = y = 0;
      }

      void
      setToOne ()
      {
        x = y = 1;
      }

      void
      setToPerpendicularVector (const Vector2& v)
      {
        *this = v;
        std::swap (x, y);
        x = -x;
      }

      T
      dot (const Vector2& v) const
      {
        return (x * v.x + y * v.y);
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
        return (lLength);
      }

      Vector2
      project (const Vector2& v, bool isVUnit = true) const
      {
        T mag = dot (v);
        if (!isVUnit)
        {
          mag *= 1 / v.lengthSquared ();
        }
        return (mag * v);
      }

      Vector2
      lerp (const Vector2& target, T targetWeight) const
      {
        return (Vector2 (x + (target.x - x) * targetWeight,
            y + (target.y - y) * targetWeight));
      }

      Vector2&
      multiplyComponents (const Vector2& v)
      {
        x *= v.x;
        y *= v.y;
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

      Vector2&
      operator+= (const Vector2& v)
      {
        x += v.x;
        y += v.y;
        return (*this);
      }

      Vector2&
      operator-= (const Vector2& v)
      {
        x -= v.x;
        y -= v.y;
        return (*this);
      }

      Vector2&
      operator*= (T scalar)
      {
        x *= scalar;
        y *= scalar;
        return (*this);
      }

      Vector2&
      operator/= (T scalar)
      {
        T invScalar = 1 / scalar;
        x *= invScalar;
        y *= invScalar;
        return (*this);
      }

    public:

      union
      {
        T coords[2];
        struct
        {
          T x, y;
        };
      };

    };

  template<typename T>
    inline
    bool
    operator== (const Vector2<T>& v1, const Vector2<T>& v2)
    {
      return (Math<T>::equalsUsingTolerance (v1.x, v2.x)
          && Math<T>::equalsUsingTolerance (v1.y, v2.y));
    }

  template<typename T>
    inline
    bool
    operator!= (const Vector2<T>& v1, const Vector2<T>& v2)
    {
      return (!(v1 == v2));
    }

  template<typename T>
    inline Vector2<T>
    operator+ (const Vector2<T>& v1, const Vector2<T>& v2)
    {
      Vector2<T> sum (v1);
      sum += v2;
      return (sum);
    }

  template<typename T>
    inline Vector2<T>
    operator- (const Vector2<T>& v1, const Vector2<T>& v2)
    {
      Vector2<T> diff (v1);
      diff -= v2;
      return (diff);
    }

  template<typename T>
    inline Vector2<T>
    operator- (const Vector2<T>& v)
    {
      return (Vector2<T> (-v.x, -v.y));
    }

  template<typename T>
    inline Vector2<T>
    multiplyComponents (const Vector2<T>& v1, const Vector2<T>& v2)
    {
      Vector2<T> prod (v1);
      prod *= v2;
      return (prod);
    }

  template<typename T>
    inline Vector2<T>
    operator* (const Vector2<T>& v, T scalar)
    {
      Vector2<T> multiple (v);
      multiple *= scalar;
      return (multiple);
    }

  template<typename T>
    inline Vector2<T>
    operator* (T scalar, const Vector2<T>& v)
    {
      return (v * scalar);
    }

  template<typename T>
    inline Vector2<T>
    operator/ (const Vector2<T>& v, T scalar)
    {
      Vector2<T> quotient (v);
      quotient /= scalar;
      return (quotient);
    }

  template<typename T>
    std::ostream&
    operator<< (std::ostream& outStream, const Vector2<T>& v)
    {
      std::ios_base::fmtflags origState = outStream.flags ();
      outStream << std::setprecision (2) << std::showpoint;
      outStream << "[ " << v.x << ", " << v.y << " ]";
      outStream.flags (origState);
      return (outStream);
    }

  typedef Vector2<float> Vector2f;
  typedef Vector2<double> Vector2d;
  typedef Vector2<int> Vector2i;
  typedef Vector2<long> Vector2l;
  typedef Vector2<uint> Vector2u;
}

#endif

