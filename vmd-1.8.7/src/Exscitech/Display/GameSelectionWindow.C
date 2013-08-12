#include <GL/glew.h>

#include <cstdio>

#include <fstream>
#include <string>

#include <QtCore/QMap>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>

#include <QtGui/QTabWidget>
#include <QtGui/QBoxLayout>
#include <QtGui/QLabel>
#include <QtGui/QMessageBox>
#include <QtGui/QScrollArea>

#include "VMDApp.h"

#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Games/GameInfoManager.hpp"
#include "Exscitech/Display/GameSelectionWindow.hpp"
#include "Exscitech/Display/GameWidgetScrollArea.hpp"

namespace Exscitech {
GameController* GameSelectionWindow::ms_gameControllerInstance =
		GameController::acquire();
std::string GameSelectionWindow::ms_gamesTabTitle = "Games";
std::string GameSelectionWindow::ms_statsTabTitle = "Statistics";

std::string GameSelectionWindow::ms_offlineModeGreeting = "VMD++ Offline Mode";

std::string GameSelectionWindow::ms_learningGamesTitle = "Learning Games";
std::string GameSelectionWindow::ms_dockingGamesTitle = "Job Submit Games";

std::string GameSelectionWindow::ms_defaultGameInfoTitle = "VMD++";
std::string GameSelectionWindow::ms_defaultGameInfoText =
		"Click on a game to see detailed instructions here";

std::string GameSelectionWindow::ms_noGameInstructionsText =
		"There are no instructions for this game";

std::string GameSelectionWindow::ms_onlineLogoutText = "Logout";
std::string GameSelectionWindow::ms_offlineLogoutText = "Return to Menu";

int GameSelectionWindow::ms_gameListsGapSize = 50;

GameSelectionWindow::GameSelectionWindow(QWidget* parent) :
		QWidget(parent), m_selectedGameWidget(NULL) {
	this->setWindowTitle(QString(ms_gameControllerInstance->m_applicationTitle.c_str()));
	this->setMinimumWidth(900);
	this->setMinimumHeight(600);
	this->move(200, 200);

	// set text based on logged in/offline status
	QString greetingLabelText;
	QString logoutButtonText;
	if (ms_gameControllerInstance->inOnlineMode()) {
		greetingLabelText = "Welcome, "
				+ QString(ms_gameControllerInstance->m_username.c_str());
		logoutButtonText = QString(ms_onlineLogoutText.c_str());
	} else {
		greetingLabelText = QString(ms_offlineModeGreeting.c_str());
		logoutButtonText = QString(ms_offlineLogoutText.c_str());
	}

	// layout for entire window
	QVBoxLayout* windowLayout = new QVBoxLayout();

	// construct greeting label (upper rh corner)
	QLabel* greetingLabel = new QLabel(greetingLabelText);
	windowLayout->addWidget(greetingLabel, 0, Qt::AlignRight);

	// create tabbed pane, to hold main contents
	QTabWidget* tabbedPane = new QTabWidget();
	tabbedPane->setMaximumWidth(16777215);

	QWidget* gamesTab = constructGameSelectionTab();
	tabbedPane->addTab(gamesTab, QString(ms_gamesTabTitle.c_str()));

	if (ms_gameControllerInstance->inOnlineMode()) {
		QWidget* statsTab = constructStatisticsTab();
		tabbedPane->addTab(statsTab, QString(ms_statsTabTitle.c_str()));
	} else {
		int statTabId = tabbedPane->addTab(new QWidget(),
				QString(ms_statsTabTitle.c_str()));
		tabbedPane->setTabEnabled(statTabId, false);
	}

	windowLayout->addWidget(tabbedPane, 100);

	// logout button (bottom lh corner)
	QPushButton* logoutReturnButton = new QPushButton(logoutButtonText);
	QObject::connect(logoutReturnButton, SIGNAL(clicked()), this,
			SLOT(logoutAndReturn()));
	windowLayout->addWidget(logoutReturnButton, 0, Qt::AlignLeft);

	// set layout
	this->setLayout(windowLayout);
}

//****************************************************************************************

void GameSelectionWindow::handleGameSelection() {
	GameChoiceWidget* newlySelected = (GameChoiceWidget*) QObject::sender();

	if (m_selectedGameWidget == NULL
			|| m_selectedGameWidget->getGameId()
					!= newlySelected->getGameId()) {
		if (m_selectedGameWidget != NULL)
			m_selectedGameWidget->drawUnselected();
		m_selectedGameWidget = newlySelected;
		m_selectedGameWidget->drawSelected();

		GameController::ExscitechGame game = m_selectedGameWidget->getGameId();

		std::string newGameTitle = GameInfoManager::getGameTitle(game);
		m_gameInfoTitle->setText(newGameTitle.c_str());

		QString gameInstructions = "";
		std::string newGameInstructionsPath =
				GameInfoManager::getGameInstructionsPath(game);

		std::ifstream inFile(newGameInstructionsPath.c_str());
		if (inFile) {
			std::string fileString((std::istreambuf_iterator<char>(inFile)),
					std::istreambuf_iterator<char>());
			gameInstructions.append(fileString.c_str());
		} else {
			gameInstructions.append(ms_noGameInstructionsText.c_str());
		}

		m_gameInfoInstructionDisplay->setHtml(gameInstructions);

		m_gamePlayButton->setEnabled(true);
	}
}

void GameSelectionWindow::playSelectedGame() {
	m_selectedGameWidget->doDoubleClick();
}

void GameSelectionWindow::logoutAndReturn() {
	bool logoutAndReturn = true;
	if (ms_gameControllerInstance->inOnlineMode()) {
		QMessageBox logoutPrompt;
		logoutPrompt.setText(
				"Do you want to logout and return to the main menu?");
		logoutPrompt.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
		logoutPrompt.setDefaultButton(QMessageBox::Yes);
		logoutPrompt.setIcon(QMessageBox::Question);
		int userChoice = logoutPrompt.exec();
		logoutAndReturn = (userChoice == QMessageBox::Yes);
	}

	if (logoutAndReturn) {
		// TODO: log out user & return to login screen
		this->hide();
		ms_gameControllerInstance->showLoginWindow();
	}
}

//****************************************************************************************

QWidget*
GameSelectionWindow::constructGameSelectionTab() {
	QWidget* gameSelectionTab = new QWidget();

	QHBoxLayout* overallTabLayout = new QHBoxLayout();

	QLayout* gameDisplayLayout = createGameDisplayArea();
	overallTabLayout->addLayout(gameDisplayLayout, 9);

	overallTabLayout->addSpacing(ms_gameListsGapSize);

	QLayout* gameInfoScreenLayout = createGameInfoArea();
	overallTabLayout->addLayout(gameInfoScreenLayout, 6);

	gameSelectionTab->setLayout(overallTabLayout);

	return (gameSelectionTab);
}

QLayout*
GameSelectionWindow::createGameDisplayArea() {
	//  set up game choice widgets

	// maps to hold widgets, ensures alphabetical ordering
	QMap<QString, GameChoiceWidget*> learningGamesMap;
	QMap<QString, GameChoiceWidget*> dockingGamesMap;

	// add each available game to the appropriate list for display
	for (int i = 0; i < ms_gameControllerInstance->NUM_GAMES; ++i) {
		GameController::ExscitechGame game = (GameController::ExscitechGame) i;
		GameChoiceWidget* gameWidget = new GameChoiceWidget(game);

		if (!ms_gameControllerInstance->inOnlineMode()
				&& !GameInfoManager::gameHasOfflineMode(game)) {
			gameWidget->setEnabled(false);
		}

		QObject::connect(gameWidget, SIGNAL(gameChoiceSelected()), this,
				SLOT(handleGameSelection()));

		GameInfoManager::GameType gameType = GameInfoManager::getGameType(game);
		QString gameTitle(GameInfoManager::getGameTitle(game).c_str());
		if (gameType == GameInfoManager::LEARNING_GAME) {
			learningGamesMap.insert(gameTitle, gameWidget);
		} else if (gameType == GameInfoManager::JOB_SUBMIT_GAME) {
			dockingGamesMap.insert(gameTitle, gameWidget);
		}
	}

	// set up gui to hold game choice widgets

	// create layout for entire game selection area
	// area will consist of game type labels and lists of games in scroll panes
	QVBoxLayout* gameSelectionArea = new QVBoxLayout();

	// construct and add 'Learning Games' label
	QLabel* learningGameLabel = new QLabel(tr(ms_learningGamesTitle.c_str()));
	learningGameLabel->setFont(QFont("Times", 14, QFont::Bold));
	gameSelectionArea->addWidget(learningGameLabel, 0, Qt::AlignLeft);

	// construct scroll pane for list of learning games
	GameWidgetScrollArea* learningGamesScrollArea = new GameWidgetScrollArea(
			learningGamesMap);
	gameSelectionArea->addWidget(learningGamesScrollArea, 1);

	// add space between learning and job submit games
	gameSelectionArea->addSpacing(ms_gameListsGapSize);

	// construct and add 'Job Submit Games' label
	QLabel* dockingGameLabel = new QLabel(tr(ms_dockingGamesTitle.c_str()));
	dockingGameLabel->setFont(QFont("Times", 14, QFont::Bold));
	gameSelectionArea->addWidget(dockingGameLabel, 0, Qt::AlignLeft);

	// construct scroll pane for list of job submit games
	GameWidgetScrollArea* dockingGamesScrollArea = new GameWidgetScrollArea(
			dockingGamesMap);
	gameSelectionArea->addWidget(dockingGamesScrollArea, 1);

	return (gameSelectionArea);
}

QLayout*
GameSelectionWindow::createGameInfoArea() {
	QVBoxLayout* gameInfoDisplay = new QVBoxLayout();

	m_gameInfoTitle = new QLabel(tr(ms_defaultGameInfoTitle.c_str()));
	gameInfoDisplay->addWidget(m_gameInfoTitle, 0, Qt::AlignCenter);

	m_gameInfoInstructionDisplay = new QWebView();
	QString defaultHtmlText = "<p>" + tr(ms_defaultGameInfoText.c_str())
			+ "</p>";
	m_gameInfoInstructionDisplay->setHtml(defaultHtmlText);
	gameInfoDisplay->addWidget(m_gameInfoInstructionDisplay, 1,
			Qt::AlignCenter);

	m_gamePlayButton = new QPushButton(tr("Play"));
	m_gamePlayButton->setDisabled(true);
	QObject::connect(m_gamePlayButton, SIGNAL(clicked()), this,
			SLOT(playSelectedGame()));
	gameInfoDisplay->addWidget(m_gamePlayButton, 0, Qt::AlignCenter);

	return (gameInfoDisplay);
}

QWidget*
GameSelectionWindow::constructStatisticsTab() {
	return (new QWidget());
}

void GameSelectionWindow::closeEvent(QCloseEvent* event) {
	// TODO: switch back
	// QMessageBox exitPrompt;
	// exitPrompt.setText("Are you sure you want to quit?");
	// exitPrompt.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
	// exitPrompt.setDefaultButton(QMessageBox::Ok);
	// exitPrompt.setIcon(QMessageBox::Question);
	// int confirmClose = exitPrompt.exec();
	// if(confirmClose == QMessageBox::Ok)

	bool confirmClose = true;
	if (confirmClose) {
		ms_gameControllerInstance->terminateApplication();
		event->accept();
	} else {
		event->ignore();
	}
}
}

