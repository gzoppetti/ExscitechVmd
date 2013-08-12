#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include <GL/glew.h>

#include "Exscitech/Graphics/Transformable.hpp"
#include "Exscitech/Math/Matrix4x4.hpp"
#include "Exscitech/Math/Vector2.hpp"
#include "Exscitech/Types.hpp"

namespace Exscitech
{
  class Camera : public Transformable
  {
  public:

    Camera (const Vector4i& viewport, float fov, float nearClip, float farClip);

    Vector2f
    getNearFar() const;

    Matrix4x4f
    getViewProjection () const;

    Matrix4x4f
    getView () const;

    Matrix4x4f
    getProjection () const;

    void
    updateView ();

    void
    setViewport (int lowerLeftX, int lowerLeftY, unsigned int width, unsigned int height);

    void
    setProjection (Single verticalFieldOfView, Single nearPlaneZ,
        Single farPlaneZ);

    void
    getProjectionVariables (float& fov, float& near, float& far);

    void
    updateProjection ();

    void
    syncVmdCamera ();

  private:

    Single m_verticalFov;
    Single m_nearPlaneZ;
    Single m_farPlaneZ;

    Vector2i m_viewportOrigin;
    int m_viewportWidth;
    int m_viewportHeight;
    Matrix4x4f m_projection;

  };
}
#endif
