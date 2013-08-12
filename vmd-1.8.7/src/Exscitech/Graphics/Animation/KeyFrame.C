#include "Exscitech/Graphics/Animation/KeyFrame.hpp"

namespace Exscitech
{
  KeyFrame::KeyFrame () :
    m_position (0.0f), m_orientation (), m_startTime (0.0f)
  {
  }

  KeyFrame::KeyFrame (const Vector3f& position, const Quaternion& orientation,
      float startTime) :
    m_position (position), m_orientation (orientation), m_startTime (startTime)
  {
  }

  void
  KeyFrame::set (const Vector3f& position, const Quaternion& orientation,
      float startTime)
  {
    m_position = position;
    m_orientation = orientation;
    m_startTime = startTime;
  }

  Vector3f
  KeyFrame::getPosition () const
  {
    return (m_position);
  }

  Quaternion
  KeyFrame::getOrientation () const
  {
    return (m_orientation);
  }

  float
  KeyFrame::getStartTime () const
  {
    return (m_startTime);
  }

  /*
   * This method will return a KeyFrame with the position and
   * orientation between the two KeyFrames at the given point in time.
   * The timeDelta must be between the start times of both frames.
   */
  KeyFrame
  KeyFrame::interpolate (const KeyFrame& nextKey, float timeDelta) const
  {
    float destinationWeight = timeDelta / (nextKey.m_startTime - m_startTime);
    Vector3f interpolatedPos = m_position.lerp (nextKey.m_position,
        destinationWeight);
    Quaternion interpolatedOrientation = m_orientation.lerp (
        nextKey.m_orientation, destinationWeight);
    interpolatedOrientation.normalize ();
    return (KeyFrame (interpolatedPos, interpolatedOrientation, timeDelta));
  }
}
