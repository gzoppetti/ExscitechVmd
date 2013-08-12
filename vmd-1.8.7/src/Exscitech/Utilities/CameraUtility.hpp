#ifndef CAMERA_UTILITY_HPP_
#define CAMERA_UTILITY_HPP_

#include "DisplayDevice.h"
#include "Matrix4.h"

#include "Exscitech/Math/Vector3.hpp"
#include "Exscitech/Math/Vector2.hpp"
#include "Exscitech/Math/Matrix3x4.hpp"
#include "Exscitech/Math/Matrix4x4.hpp"

namespace Exscitech {

class GameController;
class CameraUtility {
public:

	static void
	setDisplayDevice(DisplayDevice* device);

	static Vector3f
	getUp();

	static Vector3f
	getRight();

	static Vector3f
	getBack();

	static Vector3f
	getForward();

	static Vector3f
	getPosition();

	static Matrix3x4f
	getTransform();

	static void
	pitch(float angleDegrees);

	static void
	yaw(float angleDegrees);

	static void
	roll(float angleDegrees);

	static void
	moveUp(float distance);

	static void
	moveDown(float distance);

	static void
	moveRight(float distance);

	static void
	moveLeft(float distance);

	static void
	moveBack(float distance);

	static void
	moveForward(float distance);

	static void
	moveWorld(const Vector3f& translation);

	static void
	setCameraOrientation(const Vector3f& up, const Vector3f& forward);

	static void
	setPosition(const Vector3f& position);

	static void
	setViewport(int width, int height);

	static void
	setProjection(float right, float left, float top, float bottom,
			float nearClip, float farClip, float vSize);

	static void
	setProjection(float rightPlane, float aspectRatio, float nearPlane,
			float farPlane);

	static void
	setProjection(float rightPlane, float aspectRatio);

	static float
	getFieldOfView();

	static float
	getNearClip();

	static float
	getFarClip();

	static void
	getFrustum(float& leftClip, float& rightClip, float& bottomClip,
			float& topClip);

	static Matrix4
	getProjection();

	static Matrix4x4f
	getProjectionMatrix4x4();

	static Vector2i
	getViewportDimensions();

	static Matrix4x4f
	getView();

	static void
	getViewport(int& xOrigin, int& yOrigin, int& width, int& height);

	static void
	rotateWorld(Single angleDegrees, const Vector3f& axis);

	static void
	rotateWorldYInPlace(Single angleDegrees);

	static void
	rotateAboutPoint(Single angleDegrees, const Vector3f& axis,
			const Vector3f& point);

	static Vector3f
	convertScreenToEyeCoords(int screenX, int screenY, float eyeZ);

	static Vector3f
	convertScreenToWorldCoords(int screenX, int screenY, float eyeZ);

	static float
	convertEyeZToNdcZ(float eyeZ);

	static float
	convertNdcZToWinZ(float ndcZ);

	static void
	printVmdProjectionValues();

private:

	CameraUtility();

private:

	static DisplayDevice* ms_displayDevice;
	static GameController* ms_gameControllerInstance;

};

}

#endif
