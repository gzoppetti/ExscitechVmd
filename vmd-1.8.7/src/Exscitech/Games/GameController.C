#include <iostream>
#include <cstdio>

#include <QtGui/QApplication>
#include <QtGui/QDesktopServices>
#include <QtCore/QDir>

#include <GL/glew.h>

#include <IL/il.h>
#include <curl/curl.h>

#include "VMDApp.h"
#include "DisplayDevice.h"

#include "Exscitech/Display/QtWindow.hpp"
#include "Exscitech/Display/QtOpenGLDisplayDevice.hpp"
#include "Exscitech/Display/LoginWindow.hpp"
#include "Exscitech/Display/GameSelectionWindow.hpp"

#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Games/JobSubmitGame/JobSubmitGame.hpp"

#include "Exscitech/Games/IdentificationGame/IdentificationGame.hpp"
#include "Exscitech/Games/LindseyGame/LindseyGame.hpp"

#include "Exscitech/Utilities/CameraUtility.hpp"

namespace Exscitech {

GameController*
GameController::acquire() {
	static GameController* instance = new GameController();
	return instance;
}

GameController::GameController() :
		m_vmdApp(NULL), m_gameControllerState(IDLE), m_currentGame(NULL), m_applicationTitle(
				"Docking at home with VMD"), m_inOnlineMode(true), m_onlineModeInvalid(
				true), m_username(""), m_password(""), m_vmdGlWindow(NULL), m_loginWindow(
				NULL), m_gameChooserWindow(NULL), m_qtApp(NULL), m_vmdShouldAutoUpdate(
				false), m_gameDataFolderName("GameData"), m_serverDataFolderName(
				"ServerCommunication")

{
}

void GameController::initPlugin(VMDApp* vmdApp, QApplication* qtApp) {
	m_vmdApp = vmdApp;
	m_qtApp = qtApp;
	DisplayDevice* display = m_vmdApp->display;
	CameraUtility::setDisplayDevice(display);
	m_vmdGlWindow = static_cast<QtOpenGLDisplayDevice*>(display)->getGlWindow();

	// set up error logging file
	// TODO: uncomment this before release
	// ErrorLog::openErrorLog ();

	showLoginWindow();

	// one-time-per-application stuff for libraries used
	ilInit();
	curl_global_init(CURL_GLOBAL_ALL);
}

std::string GameController::createExscitechDirectories() {
	// TODO: test on windows.
	// Another way to get: QDesktopServices::storageLocation(QDesktopServices::HomeLocation)
	QString directoryPath = QDir::homePath().append("/.exscitech");
	QDir directory(directoryPath);
	//if (!directory.exists()) {
		QDir().mkdir(directoryPath);
		directory.mkdir(QString::fromStdString(m_serverDataFolderName));
		directory.mkdir(QString::fromStdString(m_gameDataFolderName));
	//}
	std::string directoryPathString = directoryPath.toStdString();
	return (directoryPathString);
}

std::string GameController::getExscitechDirectory() {
	static std::string exscitechDirectory = createExscitechDirectories();
	return (exscitechDirectory);
}

std::string GameController::createGameFolder(const std::string& folderName) {
	std::string folderPathString = getExscitechDirectory().append("/").append(
			m_gameDataFolderName).append("/").append(folderName);
	QString folderPath = QString::fromStdString(folderPathString);
	if (!QDir().exists(folderPath)) {
		QDir().mkdir(folderPath);
	}
	return (folderPathString);
}

void GameController::discardGameFolder(const std::string& folderName) {
	std::string folderPath = getExscitechDirectory().append("/").append(
			m_gameDataFolderName).append(folderName);

	QDir folder(QString::fromStdString(folderPath));
	// TODO: uncomment
	//folder.removeRecursively();
}

void GameController::setOnlineMode(bool online) {
	if (online != m_inOnlineMode) {
		m_inOnlineMode = online;
		m_onlineModeInvalid = true;
	}
}

bool GameController::inOnlineMode() {
	return m_inOnlineMode;
}

bool GameController::isOnlineModeInvalid() {
	return m_onlineModeInvalid;
}

void GameController::showLoginWindow() {
	if (m_loginWindow == NULL) {
		m_loginWindow = new LoginWindow();
		m_loginWindow->resize(500, 350);
	}

	m_loginWindow->show();
	m_loginWindow->move(200, 200);
}
void GameController::showGameSelectionWindow() {
	if (isOnlineModeInvalid()) {
		if (m_gameChooserWindow != NULL) {
			delete m_gameChooserWindow;
		}
		m_gameChooserWindow = new GameSelectionWindow();
		m_gameChooserWindow->resize(m_gameChooserWindow->minimumSize());
		m_onlineModeInvalid = false;
	}

	m_gameChooserWindow->show();
	m_gameChooserWindow->move(200, 200);
}

void GameController::updatePlugin() {
	switch (m_gameControllerState) {
	case IDLE:
		// Wait for game to be constructed by menu
		break;

	case GAME_CONSTRUCTED:
		// Ensure VMD updates before initializing game
		m_gameControllerState = INITIALIZE;
		break;

	case INITIALIZE:
		initialize();
		m_gameControllerState = UPDATE_LOOP;
		break;

	case UPDATE_LOOP:
		m_currentGame->update();
		break;
	}
}

void GameController::initialize() {
	// GMZ: Create game GUI here but do not interact with VMD
	m_currentGame->initWindow();
}

void GameController::handleKeyboardInput(int keyCode) {
	if (keyCode == Qt::Key_Escape) {
		GameController::stopCurrentGame();
		//terminateApplication ();
	} else if (m_currentGame != NULL) {
		m_currentGame->handleKeyboardInput(keyCode);
	}
}

void GameController::handleKeyboardUp(int key) {
	if (m_currentGame != NULL) {
		m_currentGame->handleKeyboardUp(key);
	}
}

bool GameController::handleMouseInput(int screenX, int screenY,
		Qt::MouseButton button) {
	if (m_currentGame != NULL) {
		return m_currentGame->handleMouseInput(screenX, screenY, button);
	} else {
		return false;
	}
}

bool GameController::handleMouseMove(int screenX, int screenY) {
	if (m_currentGame != NULL) {
		return m_currentGame->handleMouseMove(screenX, screenY);
	} else {
		return false;
	}
}

bool GameController::handleMouseRelease(int screenX, int screenY,
		Qt::MouseButton button) {
	if (m_currentGame != NULL) {
		return m_currentGame->handleMouseRelease(screenX, screenY, button);
	} else {
		return false;
	}
}

bool GameController::handleMouseWheel(int delta) {
	if (m_currentGame != NULL) {
		return m_currentGame->handleMouseWheel(delta);
	} else {
		return false;
	}
}

bool GameController::handleWindowResize(int width, int height) {
	if (m_currentGame != NULL) {
		return m_currentGame->handleWindowResize(width, height);
	} else {
		return false;
	}
}

void GameController::shutdownPlugin() {
	// final, one-time cleanup for libraries used goes here
	curl_global_cleanup();

	ErrorLog::closeErrorLog();

	delete m_currentGame;
	delete m_gameChooserWindow;
}

void GameController::drawGraphics() {
	if (m_gameControllerState == UPDATE_LOOP) {
		glPushAttrib(GL_VIEWPORT_BIT);
		// We're not in any of the initial stages, so it's safe to draw
		m_currentGame->drawGameGraphics();
		glPopAttrib();
	}
}

void GameController::stopCurrentGame() {
	// get rid of current game
	delete m_currentGame;
	fprintf(stderr, "Current game deleted.\n");
	m_currentGame = NULL;

	// display the chooser again
	GameController::m_gameChooserWindow->setVisible(true);

	// return to idle state
	m_gameControllerState = IDLE;
}

void GameController::startNewGame(ExscitechGame game) {
	m_gameChooserWindow->setVisible(false);

	delete m_currentGame;
	switch (game) {
	case MOLECULE_FLASHCARDS:
		m_currentGame = new LindseyGame();
		break;

	case IDENTIFICATION_GAME:
		m_currentGame = new IdentificationGame();
		break;

	case JOB_SUBMIT_GAME:
		m_currentGame = new JobSubmitGame();
		break;
		break;

	default:
		break;
	}

	m_gameControllerState = GAME_CONSTRUCTED;
}

void GameController::terminateApplication() {
	m_vmdShouldAutoUpdate = true;
	m_vmdApp->VMDexit("", 0, 0);
}

bool GameController::shouldUpdate() {
	/*std::string s = "\n";
	 if(ms_vmdShouldAutoUpdate)
	 s.append("should auto update, ");
	 else
	 s.append("should not auto update, ");
	 if(ms_vmdGlWindow == NULL )
	 s.append("glWindow null, ");
	 else
	 {
	 s.append("glWindow not null, ");
	 if(ms_vmdGlWindow->isVisible() )
	 s.append("glWindow VISIBLE, ");
	 else
	 s.append("glWindow not visible, ");
	 }
	 fprintf(stderr, "%s", s.c_str());*/

	return (m_vmdShouldAutoUpdate
			|| (m_vmdGlWindow != NULL && m_vmdGlWindow->isVisible()));
}
}
