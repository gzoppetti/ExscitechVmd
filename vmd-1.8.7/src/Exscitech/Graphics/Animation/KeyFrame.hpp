#ifndef KEYFRAME_HPP_
#define KEYFRAME_HPP_

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Quaternion.hpp"

namespace Exscitech
{
  // Time is a real number in [0, 1]
  class KeyFrame
  {
  public:

    KeyFrame ();

    KeyFrame (const Vector3f& position, const Quaternion& orientation,
        float startTime);

    void
    set (const Vector3f& position, const Quaternion& orientation,
        float startTime);

    Vector3f
    getPosition () const;

    Quaternion
    getOrientation () const;

    float
    getStartTime () const;

    KeyFrame
    interpolate (const KeyFrame& nextKey, float timeDelta) const;

  private:

    Vector3f m_position;
    Quaternion m_orientation;
    float m_startTime;

  };
}

#endif 
