#ifndef MATH_HPP_
#define MATH_HPP_

#include <cmath>
#include <algorithm>

namespace Exscitech
{

  template<typename T>
    class Math
    {
    private:

      Math ()
      {
      }

    public:
//
//#define TOLERANCE       1e-4
//#define RADIANS_PER_DEGREE      M_PI / 180
//#define DEGREES_PER_RADIAN      180 * M_1_PI

//static const T TOLERANCE = static_cast<T> (1e-4);
//static const T RADIANS_PER_DEGREE = static_cast<T> (M_PI / 180);
//static const T DEGREES_PER_RADIAN = static_cast<T> (180 * M_1_PI);

      static const T TOLERANCE;
      static const T RADIANS_PER_DEGREE;
      static const T DEGREES_PER_RADIAN;

      static T
      clamp (T x, T lowerBound, T upperBound)
      {
        return (std::min (std::max (x, lowerBound), upperBound));
      }

      static bool
      equalsUsingTolerance (T v1, T v2, T epsilon = TOLERANCE)
      {
        return (fabs (v1 - v2) <= epsilon);
      }

      static bool
      equalsUsingRelativeTolerance (T v1, T v2, T epsilon = TOLERANCE)
      {
        T delta = fabs (v1 - v2);
        T max = std::max (fabs (v1), fabs (v2));
        return (delta <= max * epsilon || delta == 0);
      }

      // Compare floating point numbers (approximate)
      //   0 for equality (v1 and v2 are within +/- epsilon units of each other)
      //   +1 if v1 is larger
      //   -1 if v1 is smaller
      static int
      compareUsingTolerance (T v1, T v2, T epsilon = TOLERANCE)
      {
        T delta = v1 - v2;
        int r = 0;
        if (delta > epsilon)
          r = 1;
        else if (delta < -epsilon)
          r = -1;
        return (r);
      }

      static T
      maxUsingTolerance (T v1, T v2, T epsilon = TOLERANCE)
      {
        int comparison = compareUsingTolerance (v1, v2, epsilon);
        switch (comparison)
        {
          case -1:
            return v2;
            break;
          case 1:
            return v1;
          default:
            return v1;
        }
      }

      static int
      compareUsingRelativeTolerance (T v1, T v2, T epsilon = TOLERANCE)
      {
        T delta = v1 - v2;
        T scaledEpsilon = epsilon * std::max (fabs (v1), fabs (v2));
        int r = 0;
        if (delta > scaledEpsilon)
          r = 1;
        else if (delta < -scaledEpsilon)
          r = -1;
        return (r);
      }

      static T
      toRadians (T angleDegrees)
      {
        return (angleDegrees * RADIANS_PER_DEGREE);
      }

      static T
      toDegrees (T angleRadians)
      {
        return (angleRadians * DEGREES_PER_RADIAN);
      }

      static T
      lengthSquared (const T v[3])
      {
        return (dot3 (v, v));
      }

      static T
      length (const T v[3])
      {
        return (sqrt (lengthSquared (v)));
      }

      static void
      normalize (T v[3])
      {
        T invLength = 1 / length (v);
        v[0] *= invLength;
        v[1] *= invLength;
        v[2] *= invLength;
      }

      static void
      cross (T result[3], const T v1[3], const T v2[3])
      {
        result[0] = +v1[1] * v2[2] - v1[2] * v2[1];
        result[1] = -v1[0] * v2[2] + v1[2] * v2[0];
        result[2] = +v1[0] * v2[1] - v1[1] * v2[0];
      }

      static T
      dot3 (const T v1[3], const T v2[3])
      {
        return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
      }

      static T
      dot4 (const T v1[4], const T v2[4])
      {
        return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3]);
      }
    };

  template<typename T>
    const T Math<T>::TOLERANCE = (1e-4);

  template<typename T>
    const T Math<T>::RADIANS_PER_DEGREE = static_cast<T> (M_PI / 180);

  template<typename T>
    const T Math<T>::DEGREES_PER_RADIAN = static_cast<T> (180 * M_1_PI);

}

#endif
