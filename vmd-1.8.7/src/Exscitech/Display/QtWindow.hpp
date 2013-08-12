#ifndef MU_QT_WINDOW_HPP_
#define MU_QT_WINDOW_HPP_

#include <QtGui/QWidget>

namespace Exscitech {
class QtVmdGlWindow;
class GameController;
class QtWindow: public QWidget {
	//Q_OBJECT

public:

	QtWindow(QWidget* parent = NULL);

protected:

	void
	keyPressEvent(QKeyEvent* event);

	void
	keyReleaseEvent(QKeyEvent* event);

	void
	mousePressEvent(QMouseEvent* event);

	void
	mouseMoveEvent(QMouseEvent* event);

	void
	mouseReleaseEvent(QMouseEvent* event);

	void
	wheelEvent(QWheelEvent* event);

	void
	resizeEvent(QResizeEvent* event);

protected:

	virtual void
	closeEvent(QCloseEvent* event);

private:

	static GameController* ms_gameControllerInstance;
	QtVmdGlWindow* m_glWindow;

};

}

#endif
