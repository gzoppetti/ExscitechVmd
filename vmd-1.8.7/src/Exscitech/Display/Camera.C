#include <GL/glew.h>

#include "Exscitech/Display/Camera.hpp"
#include "Exscitech/Math/Matrix4x4.hpp"
#include "Exscitech/Math/Math.hpp"

#include "Exscitech/Utilities/CameraUtility.hpp"

namespace Exscitech
{
  Camera::Camera (const Vector4i& viewport, float fov, float nearClip, float farClip) :
      m_viewportOrigin(viewport.x, viewport.y), m_viewportWidth(viewport.z), m_viewportHeight(viewport.w),
      m_verticalFov(fov), m_nearPlaneZ(nearClip), m_farPlaneZ(farClip)
  {
    updateView();
    updateProjection();
  }

  Vector2f
  Camera::getNearFar() const
  {
    return Vector2f(m_nearPlaneZ, m_farPlaneZ);
  }

  Matrix4x4f
  Camera::getViewProjection () const
  {
    // Is orthonormalization necessary?
    // orthonormalize ();
    Matrix3x4f world = getTransform ();
    Matrix3x4f view (world);
    view.invertRotationMatrix ();
    Matrix4x4f viewProjection = m_projection * Matrix4x4f (view);

    return (viewProjection);
  }

  Matrix4x4f
  Camera::getView () const
  {
    // Is orthonormalization necessary?
    // orthonormalize ();
    Matrix3x4f world = getTransform ();
    Matrix3x4f view (world);
    view.invertRotationMatrix ();

    return (Matrix4x4f (view));
  }

  Matrix4x4f
  Camera::getProjection () const
  {
    return (m_projection);
  }

  void
  Camera::updateView ()
  {
    orthonormalize ();
    Matrix3x4f world = getTransform ();
    Matrix3x4f view (world);
    view.invertRotationMatrix ();
    Single viewArray[16];
    view.getAsArray (viewArray);

    glMatrixMode (GL_MODELVIEW);
    glLoadMatrixf (viewArray);
  }

  void
  Camera::setViewport (int lowerLeftX, int lowerLeftY, unsigned int width,
      unsigned int height)
  {
    m_viewportOrigin.x = lowerLeftX;
    m_viewportOrigin.y = lowerLeftY;
    m_viewportWidth = width;
    m_viewportHeight = height;
    // glViewport does not need to be called while in GL_PROJECTION mode
    glViewport (m_viewportOrigin.x, m_viewportOrigin.y, m_viewportWidth,
        m_viewportHeight);
  }

  void
  Camera::setProjection (Single verticalFieldOfView, Single nearPlaneZ,
      Single farPlaneZ)
  {
    m_verticalFov = verticalFieldOfView;
    m_nearPlaneZ = nearPlaneZ;
    m_farPlaneZ = farPlaneZ;

    updateProjection();
  }

  void
  Camera::getProjectionVariables (float& fov, float& near, float& far)
  {
    fov = m_verticalFov;
    near = m_nearPlaneZ;
    far = m_farPlaneZ;
  }

  void
  Camera::updateProjection ()
  {
    glMatrixMode (GL_PROJECTION);
    Single aspectRatio = static_cast<Single> (m_viewportWidth)
        / m_viewportHeight;
    m_projection.setToPerspectiveProjection (m_verticalFov, aspectRatio,
        m_nearPlaneZ, m_farPlaneZ);
    Single projectionArr[16];
    m_projection.getAsArray (projectionArr);
    glLoadMatrixf (projectionArr);
    glMatrixMode (GL_MODELVIEW);
  }

  void
  Camera::syncVmdCamera ()
  {
    Vector3f up = getUp ();
    Vector3f forward = getForward ();
    Vector3f position = getPosition ();

    //m_nearPlaneZ = CameraUtility::getNearClip ();
    //m_farPlaneZ = CameraUtility::getFarClip ();

    float halfVerticalFov = m_verticalFov / 2.0f;
    float aspectRatio = static_cast<float> (m_viewportWidth) / m_viewportHeight;
    float rightPlane = tan (Math<float>::toRadians (halfVerticalFov))
        * m_nearPlaneZ;
    float topPlane = rightPlane / aspectRatio;
    float vSize = topPlane * 2;

    CameraUtility::setCameraOrientation (up, forward);
    CameraUtility::setPosition (position);
    CameraUtility::setProjection (rightPlane, -rightPlane, topPlane, -topPlane,
        m_nearPlaneZ, m_farPlaneZ, vSize);
  }
}

