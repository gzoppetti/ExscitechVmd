#ifndef BEZIER_CURVE_HPP_
#define BEZIER_CURVE_HPP_

#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
  class BezierCurve
  {
  public:

    BezierCurve ()
    {
    }

    /*
     * Set the point that the curve will begin at.
     */
    void
    setBeginPoint (const Vector3f& beginPoint)
    {
      m_points[0] = beginPoint;
    }

    /*
     * Set the point that the curve will end at.
     */
    void
    setEndPoint (const Vector3f& endPoint)
    {
      m_points[3] = endPoint;
    }

    /*
     * Set the first control point.  This will affect the curvature of the curve.
     */
    void
    setControlPoint1 (const Vector3f& controlPoint1)
    {
      m_points[1] = controlPoint1;
    }

    /*
     * Set the first control point.  This will affect the curvature of the curve.
     */
    void
    setControlPoint2 (const Vector3f& controlPoint2)
    {
      m_points[2] = controlPoint2;
    }

    /*
     * returns the beginning point of the curve
     */
    Vector3f
    getBeginPoint () const
    {
      return m_points[0];
    }

    // Time must be in range [0, 1]
    // The return value is the position along the curve at the normalized time.
    Vector3f
    interpolate (float normalizedTime) const
    {
      // ((1-t) + t)^(n-i) * p_i
      float oneMinusT = 1 - normalizedTime;
      Vector3f point = oneMinusT * oneMinusT * oneMinusT * m_points[0]
          + 3 * oneMinusT * oneMinusT * normalizedTime * m_points[1]
          + 3 * oneMinusT * normalizedTime * normalizedTime * m_points[2]
          + normalizedTime * normalizedTime * normalizedTime * m_points[3];

      return (point);
    }

  private:

    Vector3f m_points[4];

  };
}

#endif 
