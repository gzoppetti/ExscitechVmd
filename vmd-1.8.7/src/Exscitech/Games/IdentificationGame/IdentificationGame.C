#include "Exscitech/Games/IdentificationGame/IdentificationGame.hpp"

#include <fstream>
#include <istream>
// for testing:
#include <cstdio>

#include <QtGui/QPushButton>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QTextEdit>
#include <QtGui/QFileDialog>
#include <QtGui/QStyle>
#include <QtCore/QFile>

#include "VMDApp.h"
#include "Axes.h"
#include "MoleculeList.h"

#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Games/GameInfoManager.hpp"
#include "Exscitech/Display/QtWindow.hpp"
#include "Exscitech/Display/VmdGlWidget.hpp"

namespace Exscitech {
int IdentificationGame::ms_initialWindowWidth = 350;
int IdentificationGame::ms_initialWindowHeight = 200;
int IdentificationGame::ms_finalWindowWidth = 500;
int IdentificationGame::ms_finalWindowHeight = 700;
int IdentificationGame::ms_quitButtonGapSize = 20;

std::string IdentificationGame::ms_choicesListFileName = "Choices.txt";
std::string IdentificationGame::ms_sequenceListFileName = "Sequence.txt";
std::string IdentificationGame::ms_moleculeFileExtension = ".pdb";

std::string IdentificationGame::ms_startGameText = "Start Game";
std::string IdentificationGame::ms_quitGameText = "Quit";
std::string IdentificationGame::ms_newGameText = "New Game";

std::string IdentificationGame::ms_typeLabelText = "Type:";
std::string IdentificationGame::ms_levelLabelText = "Level:";

std::string IdentificationGame::ms_offlineInstructions =
		"In order to play offline, you must provide a folder with the files needed to play this game. Please enter the folder path below.";
std::string IdentificationGame::ms_errorTextName = "Error_Text";
std::string IdentificationGame::ms_errorMessage =
		"The directory you have selected does not contain the files needed to play this game. Please try a different folder.";
std::string IdentificationGame::ms_textBoxHeaderText = "Folder:";
std::string IdentificationGame::ms_lineEditName = "Line_Edit";
std::string IdentificationGame::ms_browseButtonText = "browse...";

char IdentificationGame::ms_choicesListDelimiter = '|';
char IdentificationGame::ms_sequenceListDelimiter = '|';
char IdentificationGame::ms_userChoiceSequenceDelimiter = '|';

std::string IdentificationGame::ms_offlineModeScoreLabel =
		"Your sequence selection:";

//***************************************************

IdentificationGame::IdentificationGame() {
	createInitialScreen();

	m_gameWindow = NULL;
	m_currentMolecule = NULL;
	m_gameState = INITIALIZING;
}

IdentificationGame::~IdentificationGame() {
	delete m_initialWindow;
	static GameController* instance = GameController::acquire();
	instance->m_vmdGlWindow->setParent(NULL);
	instance->m_vmdGlWindow->restoreDefaults();
	delete m_gameWindow;

	if (m_currentMolecule != NULL) {
		instance->m_vmdApp->molecule_delete(m_currentMolecule->id());
	}
}

void IdentificationGame::update() {
	switch (m_gameState) {
	case DROP_MOLECULE:
		dropMolecule();
		break;
	default:
		break;
	}
}

void IdentificationGame::initWindow() {

}

void IdentificationGame::handleKeyboardInput(int keyCode) {

}

void IdentificationGame::handleKeyboardUp(int key) {

}

bool IdentificationGame::handleMouseInput(int screenX, int screenY,
		int button) {
	return (false);
}

bool IdentificationGame::handleMouseRelease(int screenX, int screenY,
		int button) {
	return (false);
}

void IdentificationGame::drawGameGraphics() {

}

//***************************************************
// private members

QHBoxLayout*
IdentificationGame::createStartCancelButtons() {
	QHBoxLayout* buttonLayout = new QHBoxLayout();

	buttonLayout->addStretch(ms_initialWindowWidth);

	QPushButton* startGameButton = new QPushButton(
			tr(ms_startGameText.c_str()));
	static GameController* instance = GameController::acquire();
	if (instance->inOnlineMode()) {
		QObject::connect(startGameButton, SIGNAL(clicked()), this,
				SLOT(requestGameData()));
	} else {
		QObject::connect(startGameButton, SIGNAL(clicked()), this,
				SLOT(readInGameData()));
	}
	buttonLayout->addWidget(startGameButton, 1);

	QPushButton* quitGameButton = new QPushButton(tr(ms_quitGameText.c_str()));
	QObject::connect(quitGameButton, SIGNAL(clicked()), m_initialWindow,
			SLOT(close()));
	buttonLayout->addWidget(quitGameButton, 1);

	buttonLayout->addStretch(ms_initialWindowWidth);

	return (buttonLayout);
}

void IdentificationGame::createInitialScreen() {
	m_initialWindow = new QtWindow();

	std::string title = GameInfoManager::getGameTitle(
			GameController::IDENTIFICATION_GAME);
	m_initialWindow->setWindowTitle(QString(title.c_str()));

	QVBoxLayout* overallLayout = new QVBoxLayout();
	static GameController* instance = GameController::acquire();
	if (instance->inOnlineMode()) {
		QHBoxLayout* typeChooserLine = new QHBoxLayout();
		QLabel* typeLabel = new QLabel();
		typeLabel->setText(QString(ms_typeLabelText.c_str()));
		QComboBox* typeChooser = new QComboBox();
		typeChooserLine->addWidget(typeLabel, 1, Qt::AlignRight);
		typeChooserLine->addWidget(typeChooser, 5);
		overallLayout->addLayout(typeChooserLine);

		QHBoxLayout* levelChooserLine = new QHBoxLayout();
		QLabel* levelLabel = new QLabel();
		levelLabel->setText(QString(ms_levelLabelText.c_str()));
		QComboBox* levelChooser = new QComboBox();
		levelChooserLine->addWidget(levelLabel, 1, Qt::AlignRight);
		levelChooserLine->addWidget(levelChooser, 5);
		overallLayout->addLayout(levelChooserLine);

		QObject::connect (typeChooser, SIGNAL (currentIndexChanged (int)), this, SLOT (populateLevelList(int)));
		constructTypeList(typeChooser);
	} else {
		// create label with instructions
		QLabel* offlineInstructionsLabel = new QLabel();
		offlineInstructionsLabel->setWordWrap(true);
		offlineInstructionsLabel->setAlignment(Qt::AlignCenter);
		offlineInstructionsLabel->setText(
				QString(ms_offlineInstructions.c_str()));
		overallLayout->addWidget(offlineInstructionsLabel);

		// create blank label, for error text
		QLabel* errorTextLabel = new QLabel();
		errorTextLabel->setWordWrap(true);
		errorTextLabel->setAlignment(Qt::AlignCenter);
		errorTextLabel->setObjectName(QString(ms_errorTextName.c_str()));
		overallLayout->addWidget(errorTextLabel);

		// create the File:  ____________  |browse...| line
		QHBoxLayout* fileSelectLayout = new QHBoxLayout();
		QLabel* textBoxHeader = new QLabel();
		textBoxHeader->setText(QString(ms_textBoxHeaderText.c_str()));
		fileSelectLayout->addWidget(textBoxHeader);
		QLineEdit* folderLine = new QLineEdit();
		folderLine->setObjectName(QString(ms_lineEditName.c_str()));
		fileSelectLayout->addWidget(folderLine);
		QPushButton* browseButton = new QPushButton();
		browseButton->setText(QString(ms_browseButtonText.c_str()));
		QObject::connect(browseButton, SIGNAL(clicked()), this,
				SLOT(displayFileChooser()));
		fileSelectLayout->addWidget(browseButton);
		overallLayout->addLayout(fileSelectLayout);
	}

	QHBoxLayout* buttonLayout = createStartCancelButtons();
	overallLayout->addLayout(buttonLayout);

	m_initialWindow->setLayout(overallLayout);
	m_initialWindow->setFixedSize(ms_initialWindowWidth,
			ms_initialWindowHeight);
	m_initialWindow->move(300, 300);

	m_initialWindow->show();
}

void IdentificationGame::constructGamePlayScreen(
		const std::vector<std::string>& choicesList) {
	// construct game window
	m_gameWindow = new QtWindow();
	std::string title = GameInfoManager::getGameTitle(
			GameController::IDENTIFICATION_GAME);
	m_gameWindow->setWindowTitle(QString(title.c_str()));
	m_gameWindow->setMinimumWidth(ms_finalWindowWidth);
	m_gameWindow->setMinimumHeight(ms_finalWindowHeight);

	// openGl widget
	QVBoxLayout* overallWindowLayout = new QVBoxLayout();
	static GameController* instance = GameController::acquire();
	overallWindowLayout->addWidget(instance->m_vmdGlWindow);
	instance->m_vmdGlWindow->setVmdRespondMouse(true);
	instance->m_vmdGlWindow->setVisible(true);

	// display buttons for choices
	QStyle* gameStyle = m_gameWindow->style();
	int horizontalButtonSpacing = gameStyle->pixelMetric(
			QStyle::PM_LayoutHorizontalSpacing);
	int sideMarginWidth = gameStyle->pixelMetric(QStyle::PM_LayoutLeftMargin)
			+ gameStyle->pixelMetric(QStyle::PM_LayoutRightMargin);
	int totalButtonRowLength = sideMarginWidth;
	QHBoxLayout* buttonRow = new QHBoxLayout();
	for (int i = 0; i < choicesList.size(); ++i) {
		QString choice(choicesList[i].c_str());
		QPushButton* choiceButton = new QPushButton(choice);
		choiceButton->setObjectName(QString::number(i));
		QObject::connect(choiceButton, SIGNAL(clicked()), this,
				SLOT(processChoice()));

		totalButtonRowLength += choiceButton->sizeHint().width()
				+ horizontalButtonSpacing;
		buttonRow->addWidget(choiceButton, 1);
		if (totalButtonRowLength > ms_finalWindowWidth) {
			buttonRow->removeWidget(choiceButton);
			delete choiceButton;
			buttonRow->addStretch(ms_finalWindowWidth);
			buttonRow->insertStretch(0, ms_finalWindowWidth);
			overallWindowLayout->addLayout(buttonRow);
			buttonRow = new QHBoxLayout();
			totalButtonRowLength = 20;
			--i;
		}
	}
	buttonRow->addStretch(ms_finalWindowWidth);
	buttonRow->insertStretch(0, ms_finalWindowWidth);
	overallWindowLayout->addLayout(buttonRow);

	// space between ordinary buttons and quit button
	overallWindowLayout->addSpacing(ms_quitButtonGapSize);

	// quit button at bottom
	QHBoxLayout* quitButtonLayout = new QHBoxLayout();
	quitButtonLayout->addStretch(ms_finalWindowWidth);
	QPushButton* quitGameButton = new QPushButton(tr(ms_quitGameText.c_str()));
	QObject::connect(quitGameButton, SIGNAL(clicked()), m_gameWindow,
			SLOT(close()));
	quitButtonLayout->addWidget(quitGameButton, 1);
	quitButtonLayout->addStretch(ms_finalWindowWidth);
	overallWindowLayout->addLayout(quitButtonLayout);

	m_gameWindow->setLayout(overallWindowLayout);
	m_gameWindow->show();
}

void IdentificationGame::constructGameScoreScreen() {
	m_gameWindow = new QtWindow();
	QVBoxLayout* overallLayout = new QVBoxLayout();
	static GameController* instance = GameController::acquire();
	if (instance->inOnlineMode()) {
		QLabel* temp = new QLabel("Scores & stuff go here!!!");
		overallLayout->addWidget(temp);
	} else {
		// create dialog showing the text version of selections, to be copied and mailed
		QLabel* offlineScoreLabel = new QLabel();
		offlineScoreLabel->setText(tr(ms_offlineModeScoreLabel.c_str()));
		overallLayout->addWidget(offlineScoreLabel);
		QTextEdit* sequenceTextBox = new QTextEdit();
		sequenceTextBox->setText(QString(m_userChoiceSequence.c_str()));
		sequenceTextBox->setReadOnly(true);
		overallLayout->addWidget(sequenceTextBox, 1);
	}

	QHBoxLayout* buttonLayout = new QHBoxLayout();
	QPushButton* newGameButton = new QPushButton(tr(ms_newGameText.c_str()));
	QObject::connect(newGameButton, SIGNAL(clicked()), this,
			SLOT(showInitialScreen()));
	buttonLayout->addWidget(newGameButton);
	QPushButton* quitGameButton = new QPushButton(tr(ms_quitGameText.c_str()));
	QObject::connect(quitGameButton, SIGNAL(clicked()), m_gameWindow,
			SLOT(close()));
	buttonLayout->addWidget(quitGameButton);
	overallLayout->addLayout(buttonLayout);

	m_gameWindow->setLayout(overallLayout);

	m_gameWindow->show();
}

void IdentificationGame::initializeVmd() {
	static GameController* instance = GameController::acquire();
	instance->m_vmdApp->axes->off();
	instance->m_vmdApp->moleculeList->set_default_representation("CPK");
}

// this starts game, regardless of mode. when call this method, assume that
// 1) valid directory is in m_moleculeFilesDiretoryPath, which can be used to access necessary pdb's
// 2) valid sequence of file numbers (as strings) are in m_moleculeSequence
// 3) text of button choices are in choicesList as strings
void IdentificationGame::startGame(
		const std::vector<std::string>& choicesList) {
	// hide the initial screen, to reuse when game ends, if needed
	m_initialWindow->hide();

	// create the game window
	constructGamePlayScreen(choicesList);

	// prepare graphics
	initializeVmd();

	// start game by loading first molecule
	m_currentMoleculeIndex = 0;
	loadNextInSequence();
}

bool IdentificationGame::directoryValid() {
	QLineEdit* directoryLineEdit = m_initialWindow->findChild<QLineEdit*>(
			QString(ms_lineEditName.c_str()));
	QString directoryPath = directoryLineEdit->text();
	m_moleculeFilesDiretoryPath = directoryPath.toLocal8Bit().data();

	return (!directoryPath.isEmpty());
}

bool IdentificationGame::readInSequence() {
	std::string fileWithMoleculeSequence;
	fileWithMoleculeSequence.append(m_moleculeFilesDiretoryPath).append("/").append(
			ms_sequenceListFileName);

	std::ifstream sequenceReader;
	sequenceReader.open(fileWithMoleculeSequence.c_str());
	if (!sequenceReader) {
		return (false);
	}

	m_moleculeSequence.clear();

	std::string moleculeFileName;
	// files names must be start with 0.pdb 1.pdb 2.pdb etc. for as many as use
	int maxFileName = 0;
	while (getline(sequenceReader, moleculeFileName, ms_sequenceListDelimiter)) {
		int fileNameAsInt = atoi(moleculeFileName.c_str());
		if (fileNameAsInt > maxFileName)
			maxFileName = fileNameAsInt;
		m_moleculeSequence.push_back(moleculeFileName);
	}
	// get rid of newline
	m_moleculeSequence.back() =
			QString(m_moleculeSequence.back().c_str()).trimmed().toLocal8Bit().data();

	// ensure that files used in sequence exist in directory
	QFile testFile;
	QString pathStarter(m_moleculeFilesDiretoryPath.c_str());
	pathStarter += "/";
	QString fileExtention(ms_moleculeFileExtension.c_str());
	for (int i = 0; i <= maxFileName; ++i) {
		testFile.setFileName(pathStarter + QString::number(i) + fileExtention);
		if (!testFile.exists()) {
			return (false);
		}
	}

	// ensure there is at least one item in the sequence
	return (m_moleculeSequence.size() > 0);
}

bool IdentificationGame::readInChoices(std::vector<std::string>& choicesList) {
	std::string fileWithGameChoices;
	fileWithGameChoices.append(m_moleculeFilesDiretoryPath).append("/").append(
			ms_choicesListFileName);

	std::ifstream choicesReader;
	choicesReader.open(fileWithGameChoices.c_str());
	if (!choicesReader) {
		return (false);
	}

	std::string choice;
	while (getline(choicesReader, choice, ms_choicesListDelimiter)) {
		choicesList.push_back(choice);
	}

	// get rid of any whitespace attached to last entry
	choicesList.back() =
			QString(choicesList.back().c_str()).trimmed().toLocal8Bit().data();

	// record number of choices to use as indicator
	m_moleculeSkippedString =
			QString::number(choicesList.size()).toLocal8Bit().data();

	return (choicesList.size() > 1);
}

void IdentificationGame::setErrorText(std::string text) {
	QLabel* errorLabel = m_initialWindow->findChild<QLabel*>(
			QString(ms_errorTextName.c_str()));
	errorLabel->setText(QString(text.c_str()));
}

void IdentificationGame::loadNextInSequence() {
	if (m_currentMoleculeIndex >= m_moleculeSequence.size()) {
		handleEndOfGame();
	} else {
		// get next molecule
		std::string moleculeToLoad = m_moleculeFilesDiretoryPath;
		moleculeToLoad.append("/");
		moleculeToLoad.append(m_moleculeSequence[m_currentMoleculeIndex++]);
		moleculeToLoad.append(ms_moleculeFileExtension);

		// load molecule
		//TODO improve this
		FileSpec spec;
		static GameController* instance = GameController::acquire();
		int currentMoleculeId = instance->m_vmdApp->molecule_load(-1,
				moleculeToLoad.c_str(), NULL, &spec);
		m_currentMolecule = instance->m_vmdApp->moleculeList->mol_from_id(
				currentMoleculeId);
		instance->m_vmdApp->scene_scale_to(0.08f);
		m_currentMolecule->set_glob_trans(0, 1.5, 0);

		// calculate drop rate
		m_moleculeDropRate = calculateMoleculeDropRate();

		// set state so molecule will animate
		m_gameState = DROP_MOLECULE;
	}

}

float IdentificationGame::calculateMoleculeDropRate() {
	int numAtoms = m_currentMolecule->nAtoms;
	float dropSpeed = 0.00001 * numAtoms;
	//fprintf(stderr, "\nwill drop %d atoms at %f", numAtoms, dropSpeed);
	return (dropSpeed);
}

void IdentificationGame::dropMolecule() {
	if (m_currentMolecule->tm.mat[13] < -4) {
		processSkip();
	} else {
		m_currentMolecule->add_glob_trans(0, -m_moleculeDropRate, 0);
	}
}

void IdentificationGame::constructTypeList(QComboBox* typeChooser) {
	// TODO: send to server, get possible game types, fill in type chooser
}

void IdentificationGame::handleEndOfGame() {
	// remove the game window (remove the vmd window first, to make sure it doesn't get destroyed too)
	static GameController* instance = GameController::acquire();
	instance->m_vmdGlWindow->setParent(NULL);
	delete m_gameWindow;

	if (instance->inOnlineMode()) {
		// send to server, get scores
	}

	constructGameScoreScreen();
}

void IdentificationGame::processSkip() {
	m_gameState = PROCESS_CHOICE;
	static GameController* instance = GameController::acquire();
	instance->m_vmdApp->molecule_delete(m_currentMolecule->id());

	std::string choiceId = m_moleculeSkippedString;
	choiceId.push_back(ms_userChoiceSequenceDelimiter);
	m_userChoiceSequence.append(choiceId);

	loadNextInSequence();
}

//***************************************************
// button callbacks members

void IdentificationGame::populateLevelList(int typeId) {
	// TODO: send to server, get available levels for given type id (can do string too)
}

void IdentificationGame::displayFileChooser() {
	setErrorText("");

	QFileDialog fileChooser(m_initialWindow);
	fileChooser.setFileMode(QFileDialog::Directory);
	fileChooser.setOption(QFileDialog::ShowDirsOnly, true);

	if (fileChooser.exec()) {
		QStringList selectedFiles = fileChooser.selectedFiles();
		QString directoryPath = selectedFiles.at(0);

		QLineEdit* directoryLine = m_initialWindow->findChild<QLineEdit*>(
				QString(ms_lineEditName.c_str()));
		directoryLine->setText(directoryPath);
	}
}

void IdentificationGame::requestGameData() {
	// send level and type to server, and go from there....
}

void IdentificationGame::readInGameData() {
	std::vector<std::string> choiceNames;

	bool dataSuccessfullyRead = directoryValid() && readInSequence()
			&& readInChoices(choiceNames);
	if (dataSuccessfullyRead) {
		// start game
		startGame(choiceNames);
	} else {
		// display error message
		setErrorText(ms_errorMessage);
	}

}

void IdentificationGame::processChoice() {
	if (m_gameState == DROP_MOLECULE) {
		m_gameState = PROCESS_CHOICE;
		static GameController* instance = GameController::acquire();
		instance->m_vmdApp->molecule_delete(m_currentMolecule->id());

		QObject* buttonSelected = QObject::sender();
		std::string choiceId =
				buttonSelected->objectName().toLocal8Bit().data();
		choiceId.push_back(ms_userChoiceSequenceDelimiter);
		m_userChoiceSequence.append(choiceId);

		loadNextInSequence();
	}
}

void IdentificationGame::showInitialScreen() {
	delete m_gameWindow;

	m_userChoiceSequence = "";

	m_initialWindow->show();
}

}
