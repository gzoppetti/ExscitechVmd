#include <GL/glew.h>
#include "VMDApp.h"

#include "Exscitech/Games/GameController.hpp"

#include "Exscitech/Utilities/CameraUtility.hpp"

namespace Exscitech {
DisplayDevice* CameraUtility::ms_displayDevice = NULL;
GameController* CameraUtility::ms_gameControllerInstance =
		GameController::acquire();
CameraUtility::CameraUtility() {
}

void CameraUtility::setDisplayDevice(DisplayDevice* source) {
	ms_displayDevice = source;
}

Vector3f CameraUtility::getUp() {
	Vector3f up;
	ms_displayDevice->get_eye_up(&up[0]);
	up.normalize();
	return up;
}

Vector3f CameraUtility::getRight() {
	Vector3f up = getUp();
	Vector3f back = getBack();
	Vector3f right = up.cross(back);

	return right;
}

Vector3f CameraUtility::getBack() {
	Vector3f back = getForward();
	back.negate();
	return back;
}

Vector3f CameraUtility::getForward() {
	Vector3f forward;
	ms_displayDevice->get_eye_dir(&forward[0]);
	// Eye direction from VMD is typically not normalized!
	forward.normalize();
	return forward;
}

Vector3f CameraUtility::getPosition() {
	Vector3f translation;
	ms_displayDevice->get_eye_pos(&translation[0]);
	return translation;
}

Matrix3x4f CameraUtility::getTransform() {
	Matrix3x4f transform(getRight(), getUp(), getBack(), getPosition());

	return transform;
}

void CameraUtility::pitch(float angleDegrees) {
	Vector3f right = getRight();
	Vector3f up = getUp();
	Vector3f forward = getForward();

	Matrix4 currentTransform;
	currentTransform.rotate_axis(&right[0], angleDegrees);

	currentTransform.multpoint3d(&up[0], &up[0]);
	currentTransform.multpoint3d(&forward[0], &forward[0]);

	ms_displayDevice->set_eye_up(&up[0]);
	ms_displayDevice->set_eye_dir(&forward[0]);
}

void CameraUtility::yaw(float angleDegrees) {
	Vector3f up = getUp();
	Vector3f forward = getForward();

	Matrix4 currentTransform;
	currentTransform.rotate_axis(&up[0], angleDegrees);
	currentTransform.multpoint3d(&forward[0], &forward[0]);

	ms_displayDevice->set_eye_dir(&forward[0]);
}

void CameraUtility::roll(float angleDegrees) {
	Vector3f up = getUp();
	Vector3f back = getBack();
	//Vector3f
	Matrix4 currentTransform;
	currentTransform.rotate_axis(&back[0], angleDegrees);
	currentTransform.multpoint3d(&up[0], &up[0]);

	ms_displayDevice->set_eye_up(&up[0]);
}

void CameraUtility::moveUp(float distance) {
	Vector3f position = getPosition();
	Vector3f up = getUp();
	Vector3f change = up * distance;
	position += change;
	setPosition(position);
}

void CameraUtility::moveDown(float distance) {
	Vector3f position = getPosition();
	Vector3f up = getUp();
	Vector3f change = up * -distance;
	position += change;
	setPosition(position);
}

void CameraUtility::moveRight(float distance) {
	Vector3f position = getPosition();
	Vector3f right = getRight();
	Vector3f change = right * distance;
	position += change;
	setPosition(position);
}

void CameraUtility::moveLeft(float distance) {
	Vector3f position = getPosition();
	Vector3f right = getRight();
	Vector3f change = right * -distance;
	position += change;
	setPosition(position);
}

void CameraUtility::moveBack(float distance) {
	Vector3f position = getPosition();
	Vector3f forward = getForward();
	Vector3f change = forward * -distance;
	position += change;
	setPosition(position);
}

void CameraUtility::moveForward(float distance) {
	Vector3f position = getPosition();
	Vector3f forward = getForward();
	Vector3f change = forward * distance;
	position += change;
	setPosition(position);
}

void CameraUtility::moveWorld(const Vector3f& translation) {
	Vector3f position = getPosition();
	position += translation;
	setPosition(position);
}

void CameraUtility::setCameraOrientation(const Vector3f& up,
		const Vector3f& forward) {
	ms_displayDevice->set_eye_up(&up.x);
	ms_displayDevice->set_eye_dir(&forward.x);
}

void CameraUtility::setPosition(const Vector3f& position) {
	ms_displayDevice->set_eye_pos(&position[0]);
}

void CameraUtility::setViewport(int width, int height) {
	ms_displayDevice->resize_window(width, height);
}

void CameraUtility::setProjection(float right, float left, float top,
		float bottom, float nearClip, float farClip, float vSize) {
	ms_displayDevice->setClipPlanes(right, left, top, bottom, nearClip, farClip,
			vSize);
}

void CameraUtility::setProjection(float rightPlane, float aspectRatio,
		float nearPlane, float farPlane) {
	ms_displayDevice->setClipPlanes(rightPlane, aspectRatio, nearPlane,
			farPlane);
}

void CameraUtility::setProjection(float rightPlane, float aspectRatio) {
	ms_displayDevice->setClipPlanes(rightPlane, aspectRatio);
}

float CameraUtility::getFieldOfView() {
	// TODO: Fix this by calculating FOV from VMD
	return 60.0f;
}

float CameraUtility::getFarClip() {
	return ms_displayDevice->far_clip();
}

float CameraUtility::getNearClip() {
	return ms_displayDevice->near_clip();
}

void CameraUtility::getFrustum(float& leftClip, float& rightClip,
		float& bottomClip, float& topClip) {
	ms_displayDevice->getClipPlanes(leftClip, rightClip, bottomClip, topClip);
}

Matrix4 CameraUtility::getProjection() {
	float projection[16];
	glGetFloatv(GL_PROJECTION_MATRIX, projection);
	Matrix4 projectionMatrix(projection);
	return projectionMatrix;
}

Matrix4x4f CameraUtility::getProjectionMatrix4x4() {
	float projection[16];
	glGetFloatv(GL_PROJECTION_MATRIX, projection);
	Matrix4x4f projectionMatrix(projection);
	return projectionMatrix;
}

Vector2i CameraUtility::getViewportDimensions() {
	Vector2i viewport;
	ms_gameControllerInstance->m_vmdApp->display_get_size(&viewport[0],
			&viewport[1]);
	return viewport;
}

Matrix4x4f CameraUtility::getView() {
	float view[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	Matrix4x4f viewMatrix(view);

	return viewMatrix;
}

void CameraUtility::getViewport(int& xOrigin, int& yOrigin, int& width,
		int& height) {
	xOrigin = ms_gameControllerInstance->m_vmdGlWindow->x();
	yOrigin = ms_gameControllerInstance->m_vmdGlWindow->y();
	ms_gameControllerInstance->m_vmdApp->display_get_size(&width, &height);
}

void CameraUtility::rotateWorld(Single angleDegrees, const Vector3f& axis) {
	Matrix3x3f rotation;
	rotation.setFromAngleAxis(angleDegrees, axis);
	Vector3f up = rotation * getUp();
	Vector3f forward = rotation * getForward();
	Vector3f position = rotation * getPosition();

	ms_gameControllerInstance->m_vmdApp->display->set_eye_dir(&forward[0]);
	ms_gameControllerInstance->m_vmdApp->display->set_eye_up(&up[0]);
	ms_gameControllerInstance->m_vmdApp->display->set_eye_pos(&position[0]);
}

void CameraUtility::rotateWorldYInPlace(Single angleDegrees) {
	Matrix3x3f rotation;
	rotation.setToRotationY(angleDegrees);
	Vector3f up = rotation * getUp();
	Vector3f forward = rotation * getForward();

	ms_gameControllerInstance->m_vmdApp->display->set_eye_dir(&forward[0]);
	ms_gameControllerInstance->m_vmdApp->display->set_eye_up(&up[0]);
}

void CameraUtility::rotateAboutPoint(Single angleDegrees, const Vector3f& axis,
		const Vector3f& point) {
	Matrix3x3f rotation;
	rotation.setFromAngleAxis(angleDegrees, axis);
	Vector3f up = rotation * getUp();
	Vector3f forward = rotation * getForward();
	Vector3f position = rotation * getPosition();

	Vector3f negatedPointRot = rotation * -point;
	position += negatedPointRot + point;

	ms_gameControllerInstance->m_vmdApp->display->set_eye_dir(&forward[0]);
	ms_gameControllerInstance->m_vmdApp->display->set_eye_up(&up[0]);
	ms_gameControllerInstance->m_vmdApp->display->set_eye_pos(&position[0]);
}

// "eyeZ" is negative for points in front of the camera
Vector3f CameraUtility::convertScreenToEyeCoords(int screenX, int screenY,
		float eyeZ) {
	int xOrigin, yOrigin, viewWidth, viewHeight;
	CameraUtility::getViewport(xOrigin, yOrigin, viewWidth, viewHeight);
	// Adjust origin
	//screenX -= xOrigin;
	//screenY -= yOrigin;

	// Screen x to NDC x
	float ndcX = -1 + (2.0f * screenX) / viewWidth;

	// Screen y needs inverted
	int screenYInv = viewHeight - screenY;
	// Screen y to NDC y
	float ndcY = -1 + (2.0f * screenYInv) / viewHeight;

	// NDC x to clip coordinate x (CC x or PROJ x)
	float projX = ndcX * -eyeZ;
	// CC x to EYE x
	float leftClip, rightClip, bottomClip, topClip;
	getFrustum(leftClip, rightClip, bottomClip, topClip);
	float nearClip = CameraUtility::getNearClip();
	float eyeX = static_cast<float>(projX * rightClip / nearClip);
	// NDC y to clip coordinate y (CC y or PROJ y)
	float projY = ndcY * -eyeZ;
	// CC y to EYE y
	float eyeY = static_cast<float>(projY * topClip / nearClip);

	return (Vector3f(eyeX, eyeY, eyeZ));
}

// "eyeZ" is negative for points in front of the camera
Vector3f CameraUtility::convertScreenToWorldCoords(int screenX, int screenY,
		float eyeZ) {
	Vector3f eyePt = convertScreenToEyeCoords(screenX, screenY, eyeZ);
	Matrix3x4f eyeToWorld = getTransform();
	return (eyeToWorld * eyePt);
}

float CameraUtility::convertEyeZToNdcZ(float eyeZ) {
	float nearClip = CameraUtility::getNearClip();
	float farClip = CameraUtility::getFarClip();
	float ndcZ = (-2 * eyeZ - (farClip + nearClip)) / (farClip - nearClip);

	return (ndcZ);
}

// OpenGL maps NDC z into range [0, 1] in window coordinates
float CameraUtility::convertNdcZToWinZ(float ndcZ) {
	return ((ndcZ + 1) / 2);
}

void CameraUtility::printVmdProjectionValues() {
	ms_displayDevice->printProjectionValues();
}

}
