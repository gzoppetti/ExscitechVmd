#include "Exscitech/Display/VmdGlWidget.hpp"

#include "Exscitech/Display/QtOpenGLDisplayDevice.hpp"
#include "Exscitech/Games/GameController.hpp"

#include "VMDApp.h"
#include "DisplayDevice.h"

#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>

#include <cstdio>

namespace Exscitech {

VmdGlWidget::VmdGlWidget(QtOpenGLDisplayDevice* displayDevice, QWidget* parent) :
		QGLWidget(QGLFormat(QGL::DepthBuffer | QGL::DoubleBuffer | QGL::Rgba),
				parent), m_vmdRespondToKeys(false), m_vmdRespondToMouse(false), m_vmdRespondToWheel(
				false), m_displayDevice(displayDevice) {
	this->setFocusPolicy(Qt::StrongFocus);
}

// set whether or not key events received by the VmdWidget
// should be sent to VMD. All key events received by the
// widget are sent on the the overall widget as well, so
// they can be handled by the game
void VmdGlWidget::setVmdRespondKeys(bool respondKeys) {
	m_vmdRespondToKeys = respondKeys;
}

// set whether or not mouse events received by the VmdWidget
// should be sent to VMD. All mouse events received by the
// widget are sent on the the overall widget as well, so
// they can be handled by the game
void VmdGlWidget::setVmdRespondMouse(bool respondMouse) {
	m_vmdRespondToMouse = respondMouse;
}

void VmdGlWidget::setVmdRespondWheel(bool respondWheel) {
	m_vmdRespondToWheel = respondWheel;
}

void VmdGlWidget::restoreDefaults() {
	m_vmdRespondToKeys = false;
	m_vmdRespondToMouse = false;
	m_vmdRespondToWheel = false;
}

void VmdGlWidget::initializeGL() {
	fprintf(stderr, "initialized\n");
}

void VmdGlWidget::resizeGL(int w, int h) {
	static GameController* instance = GameController::acquire();

	if (instance->m_vmdApp != NULL && instance->m_vmdApp->display != NULL)
		instance->m_vmdApp->display->resize_window(w, h);
	fprintf(stderr, "VmdGlWidget resize: %d %d\n", w, h);
}

void VmdGlWidget::paintGL() {
	fprintf(stderr, "paint\n");
}

void VmdGlWidget::keyPressEvent(QKeyEvent* event) {
	if (m_vmdRespondToKeys) {
		m_displayDevice->m_lastEvent = event->type();
		m_displayDevice->m_lastButton = event->key();
		m_displayDevice->m_keyboardModifiers = event->modifiers();
	}
	event->ignore();
}

void VmdGlWidget::mousePressEvent(QMouseEvent* event) {
	if (m_vmdRespondToMouse) {
		m_displayDevice->m_lastEvent = event->type();
		m_displayDevice->m_lastButton = event->button();
		m_displayDevice->m_keyboardModifiers = event->modifiers();
		m_displayDevice->m_lastMouseX = event->x();
		m_displayDevice->m_lastMouseY = event->y();
	}
	event->ignore();
}

void VmdGlWidget::mouseReleaseEvent(QMouseEvent* event) {
	if (m_vmdRespondToMouse) {
		m_displayDevice->m_lastEvent = event->type();
		m_displayDevice->m_lastButton = event->button();
		m_displayDevice->m_keyboardModifiers = event->modifiers();
		m_displayDevice->m_lastMouseX = event->x();
		m_displayDevice->m_lastMouseY = event->y();
	}
	event->ignore();
}

void VmdGlWidget::mouseMoveEvent(QMouseEvent* event) {
	if (m_vmdRespondToMouse) {
		m_displayDevice->m_keyboardModifiers = event->modifiers();
		m_displayDevice->m_lastMouseX = event->x();
		m_displayDevice->m_lastMouseY = event->y();
	}
	event->ignore();
}

void VmdGlWidget::wheelEvent(QWheelEvent* event) {
	if (m_vmdRespondToWheel) {
		m_displayDevice->m_lastEvent = event->type();
		m_displayDevice->m_lastZDelta = event->delta();
	}
	event->ignore();
}
}

