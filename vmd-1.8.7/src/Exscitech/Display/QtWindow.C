#include <GL/glew.h>
#include <cstdio>
#include <QtCore/Qt>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QApplication>
#include <QtGui/qmessagebox.h>

#include "VMDApp.h"
#include "DisplayDevice.h"

#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Display/QtWindow.hpp"
#include "Exscitech/Display/GameSelectionWindow.hpp"

namespace Exscitech {

GameController* QtWindow::ms_gameControllerInstance = GameController::acquire();

QtWindow::QtWindow(QWidget* parent) :
		QWidget(parent) {
	setWindowTitle(tr("ExSciTecH"));
	setFocusPolicy(Qt::StrongFocus);
}

void QtWindow::keyPressEvent(QKeyEvent* event) {
	ms_gameControllerInstance->handleKeyboardInput(event->key());
	event->accept();
}

void QtWindow::keyReleaseEvent(QKeyEvent* event) {
	ms_gameControllerInstance->handleKeyboardUp(event->key());
	event->accept();
}

void QtWindow::mousePressEvent(QMouseEvent* event) {
	ms_gameControllerInstance->handleMouseInput(event->x(), event->y(), event->button());

	event->accept();
}

void QtWindow::mouseMoveEvent(QMouseEvent* event) {
	ms_gameControllerInstance->handleMouseMove(event->x(), event->y());
	event->accept();
}

void QtWindow::mouseReleaseEvent(QMouseEvent* event) {
	ms_gameControllerInstance->handleMouseRelease(event->x(), event->y(), event->button());
	event->accept();
}

void QtWindow::wheelEvent(QWheelEvent* event) {
	ms_gameControllerInstance->handleMouseWheel(event->delta());
	event->accept();
}

void QtWindow::resizeEvent(QResizeEvent* event) {
	QSize newSize = event->size();
	ms_gameControllerInstance->handleWindowResize(newSize.width(), newSize.height());
	event->accept();
}

void QtWindow::closeEvent(QCloseEvent* event) {
//    QMessageBox exitPrompt;
//    exitPrompt.setText("Are you sure you want to quit?");
//    exitPrompt.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
//    exitPrompt.setDefaultButton(QMessageBox::Ok);
//    exitPrompt.setIcon(QMessageBox::Question);
//    int confirmClose = exitPrompt.exec();
//    if(confirmClose == QMessageBox::Ok)

//   bool confirmClose = true;
//  if (confirmClose)
//  {
//    ms_gameControllerInstance->ms_vmdApp->VMDexit ("", 0, 0);
//    event->accept ();
//  }
//   else
//  {
//     event->ignore ();
//    }

	ms_gameControllerInstance->stopCurrentGame();

}

}

