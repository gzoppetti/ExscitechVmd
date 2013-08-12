#include "LindseyGame.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

#include <QtCore/QObject>
#include <QtGui/QPushButton>
#include <QtGui/QBoxLayout>
#include <QtGui/QLineEdit>
#include <QtCore/QFile>
#include <QtGui/QFileDialog>
#include <QtGui/QStyle>
#include <QtGui/QFontMetrics>
#include <QtGui/QMessageBox>
#include <QtXml/QDomDocument>

#include "VMDApp.h"
#include "Scene.h"
#include "Axes.h"
#include "MoleculeList.h"

#include "Exscitech/Display/QtWindow.hpp"
#include "Exscitech/Display/WebLinkWidget.hpp"
#include "Exscitech/Utilities/TransformUtility.hpp"
#include "Exscitech/Games/GameInfoManager.hpp"

#include "Exscitech/Utilities/ServerCommunicationManager.hpp"

namespace Exscitech {

GameController* LindseyGame::ms_gameControllerInstance =
		GameController::acquire();
const std::string LindseyGame::ms_selectLabelText =
		"Select a game category to play online:";
const int LindseyGame::ms_initialWindowMiddleSpace = 50;

const std::string LindseyGame::ms_chooseOwnLabelText[] = { "Choose a package:",
		"or play a practice game with your own package:" };

const std::string LindseyGame::ms_errorTextName = "Error_Text";

const std::string LindseyGame::ms_textBoxHeaderText = "Folder:";
const std::string LindseyGame::ms_lineEditName = "Line_Edit";
const std::string LindseyGame::ms_browseButtonText = "browse...";

const std::string LindseyGame::ms_startGameText = "Start Online Game";
const std::string LindseyGame::ms_startOfflineGameText = "Start Practice Game";
const std::string LindseyGame::ms_quitGameText = "Quit";

const std::string LindseyGame::ms_errorMessage =
		"The directory you have selected does not contain the files needed to play this game. Please try a different folder.";

const std::string LindseyGame::ms_moleculeDataFileName = "index.txt";
const char LindseyGame::ms_moleculeDataDelimiter = '\t';
const char LindseyGame::ms_moleculeDataChoiceDelimiter = '|';
const std::string LindseyGame::ms_moleculeFileSubdirectory = "pdb_list";
const std::string LindseyGame::ms_imageFileSubdirectory = "image_list";
const std::string LindseyGame::ms_moleculeFileExtension = ".pdb";
const int LindseyGame::ms_numImageFileExtensions = 2;
const std::string LindseyGame::ms_imageFileExtensions[2] = { ".jpg", ".png" };

const int LindseyGame::ms_initialWindowMinWidth = 350;
const int LindseyGame::ms_initialWindowMinHeight = 175;
const int LindseyGame::ms_windowMinWidth = 712;
const int LindseyGame::ms_windowMinHeight = 712;
const int LindseyGame::ms_quitButtonGapSize = 20;

const std::string LindseyGame::ms_offlineModeTitle = " Practice Game";

const std::string LindseyGame::ms_correctMessage = "Yes! The correct answer is";
const std::string LindseyGame::ms_incorrectMessage =
		"Sorry, the answer should be";
const std::string LindseyGame::ms_skippedMessage = "Too bad! The answer was";

const std::string LindseyGame::ms_winRateLabelName = "Win_Rate";
const std::string LindseyGame::ms_gameLevelLabelName = "Game_Level";
const std::string LindseyGame::ms_scoreLabelName = "Game_Score";
const std::string LindseyGame::ms_winRateLabelText = "Win Rate: ";
const std::string LindseyGame::ms_scoreLabelText = "Current Score: ";
const std::string LindseyGame::ms_gameLevelLabelText = "Current Game Level: ";

const std::string LindseyGame::ms_moreInfoButtonText = "View More Info";
const std::string LindseyGame::ms_moreInfoButtonName = "Info_label";
const std::string LindseyGame::ms_resumeButtonTextPart1 =
		"Resume\n(automatic in ";
const std::string LindseyGame::ms_resumeButtonTextPart2 = " seconds)";
const std::string LindseyGame::ms_resumeButtonTextShort = "Resume";
const std::string LindseyGame::ms_pauseButtonText = "Pause";

const std::string LindseyGame::ms_resumeButtonName = "Resume_button";
const std::string LindseyGame::ms_pauseButtonName = "Pause_button";

const std::string LindseyGame::ms_infoAreaName = "Info_Area";

const int LindseyGame::ms_pauseBetweenInMs = 4000;
const int LindseyGame::ms_updateLabelInMs = 1000;

const int LindseyGame::ms_infoWidth = 100;
const int LindseyGame::ms_infoheight = 30;
const int LindseyGame::ms_infoOffsetFromBottom = 130;
const int LindseyGame::ms_resumeWidth = 175;
const int LindseyGame::ms_resumeHeight = 50;
const int LindseyGame::ms_resumeOffsetFromBottom = 75;

const int LindseyGame::ms_infoAreaTopSpace = 20;
const int LindseyGame::ms_infoAreaMidSpace = 50;
const int LindseyGame::ms_infoAreaWidthAllowance = 500;

const std::string LindseyGame::ms_moleculeNameDisplayName = "Name_Label";
const std::string LindseyGame::ms_moleculeTypeDisplayName = "Type_Label";
const std::string LindseyGame::ms_moleculeImageLabelName = "Image_Label";
const std::string LindseyGame::ms_moleculeWebLinkLabelName = "Web_Link";

const std::string LindseyGame::ms_moleculeWikiLinkTitle = "View Wikipedia Entry";

const std::string LindseyGame::ms_packageChooserName = "Package_Chooser";

/* Server Stuff - must match with server */
const std::string LindseyGame::ms_gameIdTag = "gameID";
const std::string LindseyGame::ms_categoryTag = "category";
const std::string LindseyGame::ms_categoryIdAttr = "id";
const std::string LindseyGame::ms_boincCategoryIdTag = "platform_name";
const std::string LindseyGame::ms_platformTag = "platform";
const std::string LindseyGame::ms_workunitSpecificNameTag = "wu_name";

const std::string LindseyGame::ms_moleculeIdentifierTag = "pdb_name";
const std::string LindseyGame::ms_userCategoryChoiceTag = "answer";
const std::string LindseyGame::ms_wasResponseCorrectTag = "correct_answer";
const std::string LindseyGame::ms_moleculeIdentifierAttr = "name";
const std::string LindseyGame::ms_choiceCorrectAffirmative = "True";
const std::string LindseyGame::ms_scoreTag = "score";
const std::string LindseyGame::ms_nextMoleculeTag = "pdb_name";
const std::string LindseyGame::ms_textDisplayTag = "text_display";
const std::string LindseyGame::ms_imageTag = "image_display";
const std::string LindseyGame::ms_webAddressTag = "url";
const std::string LindseyGame::ms_requestIdTag = "requestID";

const std::string LindseyGame::ms_categoryRequestId = "category_request";
const std::string LindseyGame::ms_nextMolRequestId = "next_molecule_request";

const std::string LindseyGame::ms_endOfGameNotice = "-1";

const std::string LindseyGame::ms_defaultQuitRequestFile =
		"./vmd-1.8.7/ExscitechResources/ServerCommunication/QuitRequest2.xml";
const std::string LindseyGame::ms_quitRequestIndicator = "0";
const std::string LindseyGame::ms_quitRequestIndicatorTag = "work_req_seconds";

const std::string LindseyGame::ms_optionsFileName = "options.txt";
const std::string LindseyGame::ms_localFilesPath =
		GameController::acquire()->createGameFolder("MoleculeFlashcards");

const std::string LindseyGame::ms_gameEndMessage =
		"Congratulations! You've finished this game with a final score of ";

LindseyGame::LindseyGame() {
	prepareToBeginGame();
}

LindseyGame::~LindseyGame() {
	prepareToQuitGame();

	ms_gameControllerInstance->m_vmdGlWindow->setParent(NULL);
	delete m_window;

	ms_gameControllerInstance->m_vmdGlWindow->restoreDefaults();
}

void LindseyGame::initWindow() {

}

void LindseyGame::handleKeyboardInput(int keyCode) {

}

void LindseyGame::handleKeyboardUp(int key) {

}

bool LindseyGame::handleMouseInput(int screenX, int screenY, int button) {
	return (false);
}

bool LindseyGame::handleMouseRelease(int screenX, int screenY, int button) {
	return (false);
}

void LindseyGame::drawGameGraphics() {
	if (m_drawText) {
		glColor3f(1, 1, 1);
		QFont* font = new QFont("Courier", 20, QFont::Light, true);
		QFontMetrics sizeTester(*font);

		int startX = (m_window->width() / 2)
				- (sizeTester.width(QString(m_textToDraw.c_str())) / 2);
		int startY = ms_gameControllerInstance->m_vmdGlWindow->y() + 20;
		ms_gameControllerInstance->m_vmdGlWindow->renderText(startX, startY,
				m_textToDraw.c_str(), *font);

		font->setPointSize(25);
		startY = ms_gameControllerInstance->m_vmdGlWindow->y() + 60;
		std::string correctTypeName =
				m_moleculeChoiceTypes[m_moleculesList[m_currentMoleculeIndex].molType];
		ms_gameControllerInstance->m_vmdGlWindow->renderText(startX, startY,
				correctTypeName.c_str(), *font);
	}
}

void LindseyGame::update() {
	if (m_shouldAnimateMol) {
		animateMolecule();
	}
}

void LindseyGame::prepareToBeginGame() {
	// all game stuff
	GameController::acquire()->m_vmdApp->menu_show("main", 1);
	showInitialWindow();

	m_currentGameLevel = 1;
	m_numRight = 0;
	m_numWrong = 0;

	m_shouldAnimateMol = false;
	m_window = NULL;

	m_currentVmdMolecule = NULL;

	// offline game stuff
	m_drawText = false;

	// online game stuff
	// reset because we will use boinc requests
	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();

	instance->resetSequentialNumber();
}

void LindseyGame::showInitialWindow() {
	m_initialWindow = new QtWindow();

	std::string title = GameInfoManager::getGameTitle(
			GameController::MOLECULE_FLASHCARDS);
	m_initialWindow->setWindowTitle(QString(title.c_str()));

	QVBoxLayout* overallLayout = new QVBoxLayout();

	if (ms_gameControllerInstance->inOnlineMode()) {
//      CATEGORY/PACKAGE SELECTION CODE
//      // create label for package chooser
//      QLabel* selectLabel = new QLabel ();
//      selectLabel->setText (QString (ms_selectLabelText.c_str ()));
//      overallLayout->addWidget (selectLabel, 1, Qt::AlignLeft);
//
//      // create package chooser
//      QComboBox* packageChooser = createOnlinePackageList ();
//      overallLayout->addWidget (packageChooser, 5);

		// create button to start online game
		QHBoxLayout* buttonLayout = new QHBoxLayout();
		buttonLayout->addStretch(500);
		QPushButton* startOnlineGameButton = new QPushButton(
				tr(ms_startGameText.c_str()));
		QObject::connect(startOnlineGameButton, SIGNAL(clicked()), this,
				SLOT(initOnlineGame()));
		buttonLayout->addWidget(startOnlineGameButton, 1);
		buttonLayout->addStretch(500);
		overallLayout->addLayout(buttonLayout);

//      // spacing so things look nice
//      overallLayout->addSpacing (ms_initialWindowMiddleSpace);
	}

	// create label for choosing own package
	QLabel* browseLabel = new QLabel();
	QString labelText(
			ms_chooseOwnLabelText[ms_gameControllerInstance->inOnlineMode()].c_str());
	browseLabel->setText(labelText);
	overallLayout->addWidget(browseLabel, 1, Qt::AlignLeft);

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

	// create Start & Quit buttons
	QHBoxLayout* buttonLayout = new QHBoxLayout();
	buttonLayout->addStretch(500);

	QPushButton* startGameButton = new QPushButton(
			tr(ms_startOfflineGameText.c_str()));
	QObject::connect(startGameButton, SIGNAL(clicked()), this,
			SLOT(readGameData()));
	buttonLayout->addWidget(startGameButton, 1);

	QPushButton* quitGameButton = new QPushButton(tr(ms_quitGameText.c_str()));
	QObject::connect(quitGameButton, SIGNAL(clicked()), m_initialWindow,
			SLOT(close()));
	buttonLayout->addWidget(quitGameButton, 1);

	buttonLayout->addStretch(500);
	overallLayout->addLayout(buttonLayout);

	m_initialWindow->setLayout(overallLayout);
	m_initialWindow->setMinimumSize(ms_initialWindowMinWidth,
			ms_initialWindowMinHeight);
	m_initialWindow->move(300, 300);

	m_initialWindow->show();
}

// CATEGORY/PACKAGE SELECTION CODE
//  QComboBox*
//  LindseyGame::createOnlinePackageList ()
//  {
//    QComboBox* packageChooser = new QComboBox ();
//    packageChooser->setObjectName (QString (ms_packageChooserName.c_str ()));
//
//    std::vector<ServerCommunicationManager::ReqData> packageRequestData;
//    packageRequestData.push_back (
//        std::make_pair (
//            ms_gameIdTag,
//            ServerCommunicationManager::gameIds[GameController::MOLECULE_FLASHCARDS]));
//    packageRequestData.push_back (
//        std::make_pair (ms_requestIdTag, ms_categoryRequestId));
//    QDomDocument* packageResponse =
//        ServerCommunicationManager::makeLearningGameRequest (
//            packageRequestData);
//
//    if (packageResponse == NULL)
//    {
//      fprintf (stderr,
//          "Error: Molecule Flashcards: no package request response\n");
//    }
//    else
//    {
//      QDomNodeList categoryList = packageResponse->elementsByTagName (
//          QString (ms_categoryTag.c_str ()));
//      for (int i = 0; i < categoryList.count (); ++i)
//      {
//        QDomElement categoryElement = categoryList.at (i).toElement ();
//        QString categoryId = categoryElement.attribute (
//            QString (ms_categoryIdAttr.c_str ()));
//        QString categoryDescr = categoryElement.text ().trimmed ();
//
//        packageChooser->addItem (categoryDescr, categoryId);
//      }
//    }
//
//    return (packageChooser);
//  }

// Online only //
void LindseyGame::initOnlineGame() {
	if (getPackageFromServer()) {
		m_gameInOnlineMode = true;

		startGame();
	} else {
		displayCategoryLoadingError();
	}
}

// Offline only //
void LindseyGame::readGameData() {
	if (initMoleculeLists()) {
		m_gameInOnlineMode = false;

		startGame();
	} else {
		setErrorText(ms_errorMessage);
	}
}

// this method sets the m_moleculeFilesDiretoryPath, m_moleculeChoiceTypes
// and m_moleculesList members
bool LindseyGame::initMoleculeLists() {
	// get the folder the user chose
	QLineEdit* directoryLineEdit = m_initialWindow->findChild<QLineEdit*>(
			QString(ms_lineEditName.c_str()));
	QString directoryPath = directoryLineEdit->text();
	m_moleculeFilesDiretoryPath = directoryPath.toLocal8Bit().data();

	// make sure a folder was specified
	if (directoryPath.isEmpty())
		return (false);

	// get the index file
	std::string moleculeDataFile;
	moleculeDataFile.append(m_moleculeFilesDiretoryPath).append("/").append(
			ms_moleculeDataFileName);

	// make sure can read from index file
	std::ifstream moleculeReader;
	moleculeReader.open(moleculeDataFile.c_str());
	if (!moleculeReader) {
		return (false);
	}

	// remove any left-over data from previous initializations
	m_moleculeChoiceTypes.clear();
	m_moleculesList.clear();

	// read each category/molecules chunk
	std::string choiceTypeAndMolecules;
	while (getline(moleculeReader, choiceTypeAndMolecules,
			ms_moleculeDataChoiceDelimiter)) {
		std::istringstream choiceTypeProcessor(choiceTypeAndMolecules);
		std::string choiceTypeIdentifier;
		getline(choiceTypeProcessor, choiceTypeIdentifier);
		choiceTypeIdentifier =
				QString(choiceTypeIdentifier.c_str()).trimmed().toLocal8Bit().data();
		int currentChoiceId = m_moleculeChoiceTypes.size();
		m_moleculeChoiceTypes.push_back(choiceTypeIdentifier);

		// read each of the molecules under this category
		std::string moleculeDataString;
		while (getline(choiceTypeProcessor, moleculeDataString)) {
			std::istringstream molInfoProcessor(moleculeDataString);

			MolData moleculeData;
			getline(molInfoProcessor, moleculeData.molName,
					ms_moleculeDataDelimiter);
			moleculeData.molType = currentChoiceId;
			getline(molInfoProcessor, moleculeData.molWebPath,
					ms_moleculeDataDelimiter);

			m_moleculesList.push_back(moleculeData);
		}

	}

	// ensure that files used in sequence exist in directory
	QFile testFile;
	QString pathStarter(m_moleculeFilesDiretoryPath.c_str());
	pathStarter += "/";

	QString moleculePathStarter = pathStarter;
	moleculePathStarter += QString(ms_moleculeFileSubdirectory.c_str());
	moleculePathStarter += "/";

	QString imagePathStarter = pathStarter;
	imagePathStarter += QString(ms_imageFileSubdirectory.c_str());
	imagePathStarter += "/";

	QString moleculeFileExtention(ms_moleculeFileExtension.c_str());
	for (int i = 0; i < m_moleculesList.size(); ++i) {
		// test pdb's
		testFile.setFileName(
				moleculePathStarter + QString::number(i)
						+ moleculeFileExtention);
		if (!testFile.exists()) {
			return (false);
		}

		//test images
		QString imageName;
		bool foundFile = false;
		for (int j = 0; j < ms_numImageFileExtensions && !foundFile; ++j) {
			imageName = imagePathStarter + QString::number(i)
					+ QString(ms_imageFileExtensions[j].c_str());
			testFile.setFileName(imageName);
			foundFile = testFile.exists();
		}
		if (foundFile)
			m_moleculesList[i].molImageFile = imageName.toLocal8Bit().data();
		else
			return (false);

	}

	// ensure there is at least one molecule to show
	return (m_moleculesList.size() > 0);

}

bool LindseyGame::getPackageFromServer() {
// CATEGORY/PACKAGE SELECTION CODE
//    QComboBox* packageChooser = m_initialWindow->findChild<QComboBox*> (
//        QString (ms_packageChooserName.c_str ()));
//    m_selectedCategoryId = packageChooser->itemData (
//        packageChooser->currentIndex ()).toString ().toStdString ();

	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();

	m_selectedCategoryId = GameInfoManager::getGameServerIdCode(
			GameController::MOLECULE_FLASHCARDS);
	m_selectedCategoryId.append("_").append(
			instance->getUserData().userCategory);

	std::vector<ServerCommunicationManager::ReqData> fieldsToChange = { {
			ms_boincCategoryIdTag, m_selectedCategoryId } };

	QDomDocument* boincResponse = instance->makeBoicRequest(fieldsToChange);
	if (boincResponse == NULL) {
		ErrorLog::logError("Molecule Flashcards: no package response");
		return false;
	}

	// get and store work unit name (aka unique package identifier) for future use
	m_packageIdentifier = WorkunitId::initFromBoincResponse(boincResponse);
//    QDomElement workunitSection = boincResponse->elementsByTagName (
//        QString (ms_workunitParentTag.c_str ())).at (0).toElement ();
//    QDomElement nameTag = workunitSection.elementsByTagName (
//        QString (ms_workunitNameTag.c_str ())).at (0).toElement ();
//    m_packageIdentifier = nameTag.text ().toStdString ();

//    if (m_packageIdentifier.empty ())
	if (!m_packageIdentifier.isValid()) {
		// TODO: uncomment
		//return false;
		m_packageIdentifier.primaryName = "foodOK_3_1345052860";
		m_packageIdentifier.quitName = "foodOK_3_1345052860_0";
	}
	fprintf(stderr, "work unit names: %s, %s\n",
	m_packageIdentifier.primaryName.c_str (),
	m_packageIdentifier.quitName.c_str ());

	// get all file_info sections, except those that contain <executable/>

	bool downloadSuccess = true;
	QDomNodeList fileInfoSections = boincResponse->elementsByTagName(
			QString(instance->m_fileInfoTag.c_str()));
	for (int i = 0; i < fileInfoSections.count() && downloadSuccess; ++i) {
		QDomElement fileInfoSection = fileInfoSections.at(i).toElement();

		int executableTagsInSection = fileInfoSection.elementsByTagName(
				QString(instance->m_fileInfoIgnoreTag1.c_str())).size();
		int uploadTagsInSection = fileInfoSection.elementsByTagName(
				QString(instance->m_fileInfoIgnoreTag2.c_str())).size();
		// only process file_info sections that do not contain the executable tag
		if (executableTagsInSection == 0 && uploadTagsInSection == 0) {
			// get file data
			std::string name =
					fileInfoSection.namedItem(
							QString(instance->m_fileInfoNameTag.c_str())).toElement().text().toStdString();
			std::string fileDownloadAddress =
					fileInfoSection.namedItem(
							QString(instance->m_fileInfoUrlTag.c_str())).toElement().text().toStdString();
			std::string fileChecksum =
					fileInfoSection.namedItem(
							QString(instance->m_fileInfoChecksumTag.c_str())).toElement().text().toStdString();

			fprintf(stderr, "---->download from %s\n",
			fileDownloadAddress.c_str ());

			// download file
			std::string newFileName = ms_localFilesPath;
			newFileName.append("/").append(name);

			static ServerCommunicationManager* instance =
					ServerCommunicationManager::acquire();

			downloadSuccess = instance->downloadFile(fileDownloadAddress,
					newFileName, fileChecksum);

			// record that we have downloaded this file, to delete it when we finish
			m_filesToDeleteOnExit.push_back(name);
		}
	}
	if (!downloadSuccess) {
		ErrorLog::logError(
				"Molecule Flashcards: not all files downloaded successfully");
		return (false);
	}

	// open the options.txt file
	std::string optionsFileName = "";
	optionsFileName.append(ms_localFilesPath).append("/").append(
			ms_optionsFileName);
	std::ifstream optionFileReader;
	optionFileReader.open(optionsFileName.c_str());
	if (!optionFileReader) {
		ErrorLog::logError("Molecule Flashcards: cannot read options file %s",
				optionsFileName.c_str());
		return (false);
	}

	// category data is on first line
	std::string categoryData;
	getline(optionFileReader, categoryData);
	std::istringstream categoryLineReader(categoryData, std::istringstream::in);

	// remove any old category data
	m_moleculeChoiceTypes.clear();

	// read in category options from options file, store (for use in creating window)
	std::string categoryText;
	while (getline(categoryLineReader, categoryText,
			ms_moleculeDataChoiceDelimiter)
			&& !QString(categoryText.c_str()).trimmed().isEmpty()) {

		// extract id number and category name
		QString categoryLineManipulator(categoryText.c_str());
		categoryLineManipulator = categoryLineManipulator.trimmed();
		int spaceIndex = categoryLineManipulator.indexOf(' ');
		int categoryIdNumber = categoryLineManipulator.left(spaceIndex).toInt();
		categoryText =
				categoryLineManipulator.mid(spaceIndex).trimmed().toStdString();

		// make sure categories are in order
		if (categoryIdNumber != m_moleculeChoiceTypes.size()) {
			ErrorLog::logError(
					"Molecule Flashcards: category data not in order");
			return (false);
		}

		// remove excess whitespace
		categoryText = QString(categoryText.c_str()).trimmed().toStdString();
		m_moleculeChoiceTypes.push_back(categoryText);

	}

	// file for first molecule to show on next line
	getline(optionFileReader, m_currentMoleculeFile);

	// everything has gone successfully and we're ready to start a game
	return true;

}

void LindseyGame::startGame() {
	ErrorLog::logMessage("Beginning Molecule Flashcards Game");

	delete m_initialWindow;

	createGameWindow();

	ms_gameControllerInstance->m_vmdApp->axes->off();
	ms_gameControllerInstance->m_vmdApp->moleculeList->set_default_representation("CPK");
	ms_gameControllerInstance->m_vmdApp->display_set_nearclip(0.001, false);

	m_labelUpdateTimer = new QTimer(this);
	QObject::connect(m_labelUpdateTimer, SIGNAL(timeout()), this,
			SLOT(updateLabelText()));

	loadNewMolecule();
}

void LindseyGame::updateLabelText() {
	QPushButton* resumeButton = m_window->findChild<QPushButton*>(
			QString(ms_resumeButtonName.c_str()));
	int number = QString(
			resumeButton->text().at(ms_resumeButtonTextPart1.length())).toInt();
	if (number == 1) {
		m_labelUpdateTimer->stop();
		resumeGame();
	} else {
		resumeButton->setText(
				QString(ms_resumeButtonTextPart1.c_str())
						+ QString::number(number - 1)
						+ QString(ms_resumeButtonTextPart2.c_str()));
	}
}

void LindseyGame::setErrorText(const std::string& text) {
	QLabel* errorLabel = m_initialWindow->findChild<QLabel*>(
			QString(ms_errorTextName.c_str()));
	errorLabel->setText(QString(text.c_str()));
}

void LindseyGame::displayCategoryLoadingError() {
// CATEGORY/PACKAGE SELECTION CODE
//    QComboBox* packageChooser = m_initialWindow->findChild<QComboBox*> (
//        QString (ms_packageChooserName.c_str ()));
//    QString attemptedCategory = packageChooser->currentText ();

	QMessageBox notification(m_initialWindow);
//    notification.setText (
//        "Problem loading \"" + attemptedCategory
//            + "\" -- no response from server. Please try again.");
	notification.setText(
			"Problem loading online game -- incorrect response from server. Please try again.");
	notification.setStandardButtons(QMessageBox::Ok);
	notification.setIcon(QMessageBox::Warning);
	notification.exec();
}

void LindseyGame::displayFileChooser() {
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

void LindseyGame::createGameWindow() {
	m_window = new QtWindow();
	std::string title = GameInfoManager::getGameTitle(
			GameController::MOLECULE_FLASHCARDS);
	if (!m_gameInOnlineMode) {
		title.append(ms_offlineModeTitle);
	}
	m_window->setWindowTitle(QString(title.c_str()));
	m_window->setMinimumWidth(ms_windowMinWidth);
	m_window->setMinimumHeight(ms_windowMinHeight);

	QHBoxLayout* gamePlayAndInfoAreas = new QHBoxLayout();

	QVBoxLayout* overallLayout = new QVBoxLayout();

	// game state information, to be displayed continuously while user plays
	QHBoxLayout* topStuff = new QHBoxLayout();
	QLabel* gameLevelLabel = new QLabel(
			QString(ms_gameLevelLabelText.c_str())
					+ QString::number(m_currentGameLevel));
	gameLevelLabel->setObjectName(QString(ms_gameLevelLabelName.c_str()));
	gameLevelLabel->setFont(QFont("Times", 18, QFont::Bold));
	topStuff->addWidget(gameLevelLabel, 1, Qt::AlignCenter);

	if (m_gameInOnlineMode) {
		QLabel* currentScoreLabel = new QLabel(
				QString(ms_scoreLabelText.c_str()) + "0 ");
		currentScoreLabel->setObjectName(QString(ms_scoreLabelName.c_str()));
		currentScoreLabel->setFont(QFont("Times", 18, QFont::Bold));
		topStuff->addWidget(currentScoreLabel, 1, Qt::AlignCenter);
		overallLayout->addLayout(topStuff, 1);
	} else {
		QLabel* winRateLabel = new QLabel(
				QString(ms_winRateLabelText.c_str()) + "-- ");
		winRateLabel->setObjectName(QString(ms_winRateLabelName.c_str()));
		winRateLabel->setFont(QFont("Times", 18, QFont::Bold));
		topStuff->addWidget(winRateLabel, 1, Qt::AlignCenter);
		overallLayout->addLayout(topStuff, 1);
	}

	// vmd widget for displaying molecules
	overallLayout->addWidget(ms_gameControllerInstance->m_vmdGlWindow, 100);
	ms_gameControllerInstance->m_vmdGlWindow->setVisible(true);
	ms_gameControllerInstance->m_vmdGlWindow->setVmdRespondMouse(true);

	// row(s) of buttons for choices
	QStyle* gameStyle = m_window->style();
	int horizontalButtonSpacing = gameStyle->pixelMetric(
			QStyle::PM_LayoutHorizontalSpacing);
	int sideMarginWidth = gameStyle->pixelMetric(QStyle::PM_LayoutLeftMargin)
			+ gameStyle->pixelMetric(QStyle::PM_LayoutRightMargin);
	int totalButtonRowLength = sideMarginWidth;
	QHBoxLayout* buttonRow = new QHBoxLayout();
	for (int i = 0; i < m_moleculeChoiceTypes.size(); ++i) {
		QString choice(m_moleculeChoiceTypes[i].c_str());
		QPushButton* choiceButton = new QPushButton(choice);
		choiceButton->setObjectName(QString::number(i));
		QObject::connect(choiceButton, SIGNAL(clicked()), this,
				SLOT(processChoice()));

		totalButtonRowLength += choiceButton->sizeHint().width()
				+ horizontalButtonSpacing;
		buttonRow->addWidget(choiceButton, 1);
		if (totalButtonRowLength > ms_windowMinWidth) {
			buttonRow->removeWidget(choiceButton);
			delete choiceButton;
			buttonRow->addStretch(ms_windowMinWidth);
			buttonRow->insertStretch(0, ms_windowMinWidth);
			overallLayout->addLayout(buttonRow);
			buttonRow = new QHBoxLayout();
			totalButtonRowLength = 20;
			--i;
		}
	}
	buttonRow->addStretch(ms_windowMinWidth);
	buttonRow->insertStretch(0, ms_windowMinWidth);
	overallLayout->addLayout(buttonRow);

	// space between ordinary buttons and quit button
	overallLayout->addSpacing(ms_quitButtonGapSize);

	// quit button at bottom
	QHBoxLayout* quitButtonLayout = new QHBoxLayout();
	quitButtonLayout->addStretch(ms_windowMinWidth);
	QPushButton* quitGameButton = new QPushButton(tr(ms_quitGameText.c_str()));
	QObject::connect(quitGameButton, SIGNAL(clicked()), m_window,
			SLOT(close()));
	quitButtonLayout->addWidget(quitGameButton, 1);
	quitButtonLayout->addStretch(ms_windowMinWidth);
	overallLayout->addLayout(quitButtonLayout);

	gamePlayAndInfoAreas->addLayout(overallLayout, 1);

	QWidget* infoArea = createInfoArea();
	gamePlayAndInfoAreas->addWidget(infoArea, 1);
	infoArea->hide();

	m_window->setLayout(gamePlayAndInfoAreas);
	m_window->show();

	if (!m_gameInOnlineMode) {
		QPushButton* moreInfoButton = new QPushButton(
				QString(ms_moreInfoButtonText.c_str()), m_window);
		moreInfoButton->setObjectName(QString(ms_moreInfoButtonName.c_str()));
		QObject::connect(moreInfoButton, SIGNAL(clicked()), this,
				SLOT(displayMoreInfo()));
		moreInfoButton->hide();

		QPushButton* resumePlayButton = new QPushButton(m_window);
		resumePlayButton->setObjectName(QString(ms_resumeButtonName.c_str()));
		QObject::connect(resumePlayButton, SIGNAL(clicked()), this,
				SLOT(resumeGame()));
		resumePlayButton->hide();
	}
}

QWidget*
LindseyGame::createInfoArea() {
	QWidget* infoArea = new QWidget();
	infoArea->setObjectName(QString(ms_infoAreaName.c_str()));

	QVBoxLayout* overallLayout = new QVBoxLayout();

	overallLayout->addSpacing(ms_infoAreaTopSpace);

	QLabel* moleculeNameLabel = new QLabel();
	moleculeNameLabel->setObjectName(
			QString(ms_moleculeNameDisplayName.c_str()));
	moleculeNameLabel->setFont(QFont("Times", 20, QFont::Bold));
	moleculeNameLabel->setWordWrap(true);
	moleculeNameLabel->setAlignment(Qt::AlignCenter);
	moleculeNameLabel->setSizePolicy(QSizePolicy::MinimumExpanding,
			QSizePolicy::Minimum);
	overallLayout->addWidget(moleculeNameLabel, 10, Qt::AlignCenter);

	if (!m_gameInOnlineMode) {
		QLabel* moleculeTypeLabel = new QLabel();
		moleculeTypeLabel->setObjectName(
				QString(ms_moleculeTypeDisplayName.c_str()));
		moleculeTypeLabel->setFont(QFont("Times", 18, QFont::Bold));
		overallLayout->addWidget(moleculeTypeLabel, 1, Qt::AlignCenter);
	}

	overallLayout->addSpacing(ms_infoAreaMidSpace);

	QLabel* molImageLabel = new QLabel();
	molImageLabel->setObjectName(QString(ms_moleculeImageLabelName.c_str()));
	overallLayout->addWidget(molImageLabel, 10, Qt::AlignCenter);

	WebLinkWidget* wikiLinkWidget = new WebLinkWidget(ms_moleculeWikiLinkTitle);
	wikiLinkWidget->setObjectName(QString(ms_moleculeWebLinkLabelName.c_str()));
	overallLayout->addWidget(wikiLinkWidget, 10, Qt::AlignCenter);

	QPushButton* secondaryResumeButton = new QPushButton(
			QString(ms_resumeButtonTextShort.c_str()));
	if (m_gameInOnlineMode) {
		// secondary resume button is main resume button
		// (the one we get when we look for one)
		secondaryResumeButton->setObjectName(
				QString(ms_resumeButtonName.c_str()));

		QPushButton* pauseButton = new QPushButton(
				QString::fromStdString(ms_pauseButtonText));
		pauseButton->setObjectName(QString(ms_pauseButtonName.c_str()));
		QObject::connect(pauseButton, SIGNAL(clicked()), this,
				SLOT(pauseOnlineGame()));

		QHBoxLayout* buttonsLayout = new QHBoxLayout();
		buttonsLayout->addWidget(pauseButton, 1, Qt::AlignCenter);
		buttonsLayout->addWidget(secondaryResumeButton, 1, Qt::AlignCenter);

		overallLayout->addLayout(buttonsLayout, 1);
	} else {
		overallLayout->addWidget(secondaryResumeButton, 1, Qt::AlignCenter);
	}
	QObject::connect(secondaryResumeButton, SIGNAL(clicked()), this,
			SLOT(resumeGame()));

	infoArea->setLayout(overallLayout);
	return (infoArea);
}

void LindseyGame::processChoice() {
	if (m_shouldAnimateMol) {
		// stop animation
		m_shouldAnimateMol = false;

		QObject* buttonSelected = QObject::sender();
		int userChoiceType = buttonSelected->objectName().toInt();

		// remove current molecule
		ms_gameControllerInstance->m_vmdApp->molecule_delete(m_currentVmdMolecule->id());

		if (m_gameInOnlineMode) {
			sendChoiceToServer(userChoiceType);
		} else {
			// determine if user was correct
			int actualChoiceType =
					m_moleculesList[m_currentMoleculeIndex].molType;
			bool wasCorrect = (actualChoiceType == userChoiceType);

			// over-screen text
			m_textToDraw =
					(wasCorrect) ? ms_correctMessage : ms_incorrectMessage;
			m_drawText = true;

			updateSpeedLevel(wasCorrect);
			updateWinRate();

			updateButtons();

		}
	}
}

void LindseyGame::processSkip() {
	if (m_shouldAnimateMol) {
		// stop animation
		m_shouldAnimateMol = false;

		// remove current molecule
		ms_gameControllerInstance->m_vmdApp->molecule_delete(m_currentVmdMolecule->id());

		if (m_gameInOnlineMode) {
			sendChoiceToServer(-1);
		} else {
			// over-screen text
			m_textToDraw = ms_skippedMessage;
			m_drawText = true;

			updateSpeedLevel(false);
			updateWinRate();

			updateButtons();
		}
	}
}

void LindseyGame::sendChoiceToServer(int categoryChoice) {
	////////////////////////////////////
	//// prepare response to server ////
	////////////////////////////////////

	std::vector<ServerCommunicationManager::ReqData> selectionData;
	// work unit name
	selectionData.push_back(
			{ ms_workunitSpecificNameTag, m_packageIdentifier.primaryName });
	// molecule identifier for last molecule shown
	selectionData.push_back(
			{ ms_moleculeIdentifierTag, m_currentMoleculeFile });
	// user's category choice
	std::string stringCategoryChoice =
			QString::number(categoryChoice).toStdString();
	selectionData.push_back(
			{ ms_userCategoryChoiceTag, stringCategoryChoice });
	// request id
	selectionData.push_back( { ms_requestIdTag, ms_nextMolRequestId });

	/////////////////////////////////
	//// send response to server ////
	/////////////////////////////////
	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();
	QDomDocument* serverResponse = instance->makeLearningGameRequest(
			selectionData);

	///////////////////////////////////
	//// check for inconsistencies ////
	///////////////////////////////////

	// check for curl errors
	if (serverResponse == NULL) {
		ErrorLog::logError(
				"Molecule Flashcards: No game specific request response");
		alertServerError();
		return;
	}

	// test work unit name to ensure matches
	QDomElement workunitElement = serverResponse->elementsByTagName(
			QString(ms_workunitSpecificNameTag.c_str())).at(0).toElement();
	std::string responsePackageId =
			workunitElement.text().trimmed().toStdString();
	if (responsePackageId.compare(m_packageIdentifier.primaryName) != 0) {
		ErrorLog::logError(
				"Molecule Flashcards: Package identifier does not match response");
		alertServerError();
		return;
	}

	// test mol id in correct answer tag
	QDomElement correctAnswerElement = serverResponse->elementsByTagName(
			QString(ms_wasResponseCorrectTag.c_str())).at(0).toElement();
	std::string moleculeIdentifier = correctAnswerElement.attribute(
			QString(ms_moleculeIdentifierAttr.c_str())).trimmed().toStdString();
	if (moleculeIdentifier.compare(m_currentMoleculeFile) != 0) {
		ErrorLog::logError(
				"Molecule Flashcards: last shown molecule file does not match response");
		alertServerError();
		return;
	}

	/////////////////////////////////////////////////
	//// prepare window to display data received ////
	/////////////////////////////////////////////////

	// load latest molecule shown
	FileSpec spec;
	std::string fileName = ms_localFilesPath;
	fileName.append("/").append(m_currentMoleculeFile);
	int currentVmdMoleculeId = ms_gameControllerInstance->m_vmdApp->molecule_load(-1,
			fileName.c_str(), NULL, &spec);
	m_currentVmdMolecule = ms_gameControllerInstance->m_vmdApp->moleculeList->mol_from_id(
			currentVmdMoleculeId);

	// resize window
	m_window->resize(m_window->width() + ms_infoAreaWidthAllowance,
			m_window->height());

	// get the info area
	QWidget* infoArea = m_window->findChild<QWidget*>(
			QString(ms_infoAreaName.c_str()));

	// prepare text display
	QDomElement textDisplayElement = serverResponse->elementsByTagName(
			QString(ms_textDisplayTag.c_str())).at(0).toElement();
	QString displayText = textDisplayElement.text().trimmed();
	QLabel* primaryLabel = infoArea->findChild<QLabel*>(
			QString(ms_moleculeNameDisplayName.c_str()));
	primaryLabel->setText(displayText);

	// prepare image
	QLabel* molImageLabel = infoArea->findChild<QLabel*>(
			QString(ms_moleculeImageLabelName.c_str()));
	QDomElement imageElement = serverResponse->elementsByTagName(
			QString(ms_imageTag.c_str())).at(0).toElement();
	QString imagePath(ms_localFilesPath.c_str());
	imagePath += "/" + imageElement.text().trimmed();
	QPixmap moleculeImage(imagePath);
	molImageLabel->setPixmap(moleculeImage);

	// prepare web link
	WebLinkWidget* wikiLink = infoArea->findChild<WebLinkWidget*>(
			QString(ms_moleculeWebLinkLabelName.c_str()));
	QDomElement webAddrElement = serverResponse->elementsByTagName(
			QString(ms_webAddressTag.c_str())).at(0).toElement();
	std::string webPath = webAddrElement.text().trimmed().toStdString();
	wikiLink->setLinkAddress(webPath);

	// update score with score sent from server
	QDomElement scoreElement = serverResponse->elementsByTagName(
			QString(ms_scoreTag.c_str())).at(0).toElement();
	QString score = scoreElement.text().trimmed();
	QLabel* scoreLabel = m_window->findChild<QLabel*>(
			QString(ms_scoreLabelName.c_str()));
	scoreLabel->setText(QString(ms_scoreLabelText.c_str()) + score);

	//////////////////////////////////////
	//// Get other data from response ////
	//////////////////////////////////////

	// determine if user's choice was correct
	bool wasChoiceCorrect = (correctAnswerElement.text().trimmed().compare(
			QString(ms_choiceCorrectAffirmative.c_str())) != 0);
	updateSpeedLevel(wasChoiceCorrect);

	// check to make sure game is not ending
	int gameEndTags = serverResponse->elementsByTagName(
			QString(instance->m_gameEndTag.c_str())).size();
	if (gameEndTags > 0) {
		m_currentMoleculeFile = ms_endOfGameNotice;
	} else {
		// store next molecule to display file name, put in m_currentMoleculeFile
		QDomElement nextMolElement = serverResponse->elementsByTagName(
				QString(ms_nextMoleculeTag.c_str())).at(0).toElement();
		m_currentMoleculeFile = nextMolElement.text().trimmed().toStdString();
	}

	/////////////////////
	//// finalize UI ////
	/////////////////////

	// this will prepare resume button
	// depends on m_currentMoleucleFile being set to next molecule to show,
	// which is why it is here instead of with the rest of the UI modification code
	updateButtons();

	// show the info are we've set up
	infoArea->show();

}

void LindseyGame::updateSpeedLevel(bool wasCorrect) {
	// update game speed
	if (wasCorrect)
		++m_numRight;
	else
		++m_numWrong;

	bool gameLevelChanged = false;
	if (m_numRight - m_numWrong >= 5 * m_currentGameLevel) {
		++m_currentGameLevel;
		gameLevelChanged = true;
	} else if (m_numWrong > m_numRight && m_currentGameLevel != 1) {
		--m_currentGameLevel;
		gameLevelChanged = true;
	}

	if (gameLevelChanged) {
		QLabel* gameLevelLabel = m_window->findChild<QLabel*>(
				QString(ms_gameLevelLabelName.c_str()));
		gameLevelLabel->setText(
				QString(ms_gameLevelLabelText.c_str())
						+ QString::number(m_currentGameLevel));
	}
}

void LindseyGame::updateButtons() {
	int startY = ms_gameControllerInstance->m_vmdGlWindow->y()
			+ ms_gameControllerInstance->m_vmdGlWindow->height();
	if (!m_gameInOnlineMode) {
		// if in offline mode, show the "View More Info" button
		QPushButton* infoButton = m_window->findChild<QPushButton*>(
				QString(ms_moreInfoButtonName.c_str()));
		infoButton->resize(ms_infoWidth, ms_infoheight);
		int startX = m_window->width() / 2 - ms_infoWidth / 2;
		infoButton->move(startX, startY - ms_infoOffsetFromBottom);
		infoButton->show();
	}

	bool notEndOnlineGame = (!m_gameInOnlineMode
			|| m_currentMoleculeFile.compare(ms_endOfGameNotice) != 0);

	QPushButton* resumeButton = m_window->findChild<QPushButton*>(
			QString(ms_resumeButtonName.c_str()));
	QPushButton* pauseButton = m_window->findChild<QPushButton*>(
			QString(ms_pauseButtonName.c_str()));
	if (notEndOnlineGame) {
		// we're in offline mode, or we're in online mode and haven't reached the end of the game
		resumeButton->setText(
				QString(ms_resumeButtonTextPart1.c_str())
						+ QString::number(ms_pauseBetweenInMs / 1000)
						+ QString(ms_resumeButtonTextPart2.c_str()));
		resumeButton->resize(ms_resumeWidth, ms_resumeHeight);

		if (m_gameInOnlineMode) {
			pauseButton->show();
		}
	} else {
		// in online mode and there are no more molecules to display
		resumeButton->setText("This game has finished. Start new game.");
		resumeButton->resize(180, 50);
		QObject::disconnect(resumeButton, SIGNAL(clicked()), this,
				SLOT(resumeGame()));
		QObject::connect(resumeButton, SIGNAL(clicked()), this,
				SLOT(startNewGame()));

		pauseButton->hide();
	}

	if (!m_gameInOnlineMode) {
		int startX = m_window->width() / 2 - ms_resumeWidth / 2;
		resumeButton->move(startX, startY - ms_resumeOffsetFromBottom);
	}
	resumeButton->show();
	resumeButton->setFocus();
	if (notEndOnlineGame) {
		m_labelUpdateTimer->start(ms_updateLabelInMs);
	}
	//QTimer::singleShot(4000, this, SLOT(resumeGame()));
}

void LindseyGame::updateWinRate() {
	int winRate = (int) (100 * m_numRight / (float) (m_numRight + m_numWrong));
	QLabel* winRateLabel = m_window->findChild<QLabel*>(
			QString(ms_winRateLabelName.c_str()));
	winRateLabel->setText(
			QString(ms_winRateLabelText.c_str())
					+ QString::number(winRate, 'f', 2) + "%");
}

void LindseyGame::resumeGame() {
	m_drawText = false;

	m_labelUpdateTimer->stop();

	if (!m_gameInOnlineMode) {
		QPushButton* infoButton = m_window->findChild<QPushButton*>(
				QString(ms_moreInfoButtonName.c_str()));
		infoButton->hide();
		QPushButton* resumeButton = m_window->findChild<QPushButton*>(
				QString(ms_resumeButtonName.c_str()));
		resumeButton->hide();
	}
	QWidget* infoArea = m_window->findChild<QWidget*>(
			QString(ms_infoAreaName.c_str()));
	if (infoArea->isVisible()) {
		infoArea->hide();
		m_window->resize(m_window->width() - ms_infoAreaWidthAllowance,
				m_window->height());

		ms_gameControllerInstance->m_vmdApp->molecule_delete(m_currentVmdMolecule->id());
	}
	loadNewMolecule();
}

void LindseyGame::pauseOnlineGame() {
	m_labelUpdateTimer->stop();

	QPushButton* resumeButton = m_window->findChild<QPushButton*>(
			QString(ms_resumeButtonName.c_str()));
	resumeButton->setText(QString::fromStdString(ms_resumeButtonTextShort));

	QPushButton* pauseButton = m_window->findChild<QPushButton*>(
			QString(ms_pauseButtonName.c_str()));
	pauseButton->hide();
}

void LindseyGame::animateMolecule() {
	bool offScreen =
			(m_isVerticalDrop) ?
					(m_currentVmdMolecule->tm.mat[13] < -4) :
					(m_currentVmdMolecule->tm.mat[14] > 3);
	if (offScreen) {
		processSkip();
	} else {

		// account for game level & molecule size in decrement
		float decrement = -1 * m_molAnimationRates[0] * m_currentGameLevel;
		float changedY = (m_isVerticalDrop) ? decrement : 0;
		float changedZ = (m_isVerticalDrop) ? 0 : -1 * decrement;
		m_currentVmdMolecule->add_glob_trans(0, changedY, changedZ);

		TransformUtility::yaw(m_currentVmdMolecule, m_molAnimationRates[1]);
		TransformUtility::pitch(m_currentVmdMolecule, m_molAnimationRates[1]);
		TransformUtility::roll(m_currentVmdMolecule, m_molAnimationRates[1]);
	}
}

void LindseyGame::setMoleculeAnimationRate() {
	int numAtoms = m_currentVmdMolecule->nAtoms;
	float translateSpeed = 0.000005 * numAtoms;
	m_molAnimationRates[0] = translateSpeed;
	m_molAnimationRates[1] = translateSpeed * 75;
	//fprintf(stderr, "\nwill drop %d atoms at %f", numAtoms, dropSpeed);
}

void LindseyGame::loadNewMolecule() {
	m_isVerticalDrop = (rand() % 2 == 1);

	// obtain the name of the file to load
	std::string fileName;
	if (m_gameInOnlineMode) {
		fileName = ms_localFilesPath;
		fileName.append("/").append(m_currentMoleculeFile);
	} else {
		m_currentMoleculeIndex = rand() % m_moleculesList.size();

		fileName = m_moleculeFilesDiretoryPath;
		fileName.append("/").append(ms_moleculeFileSubdirectory);
		fileName.append("/").append(
				QString::number(m_currentMoleculeIndex).toLocal8Bit().data());
		fileName.append(ms_moleculeFileExtension);
	}

	FileSpec spec;
	int currentVmdMoleculeId = ms_gameControllerInstance->m_vmdApp->molecule_load(-1,
			fileName.c_str(), NULL, &spec);
	m_currentVmdMolecule = ms_gameControllerInstance->m_vmdApp->moleculeList->mol_from_id(
			currentVmdMoleculeId);

	QFile(QString(fileName.c_str())).close();

	ms_gameControllerInstance->m_vmdApp->scene_scale_to(0.08f);
	float startX = 0;
	float startY = (m_isVerticalDrop) ? 1.5f : 0;
	float startZ = (m_isVerticalDrop) ? 0 : -1.5;
	m_currentVmdMolecule->set_glob_trans(startX, startY, startZ);

	float randomXRotation = (float) (rand() % 180);
	float randomYRotation = (float) (rand() % 180);
	float randomZRotation = (float) (rand() % 180);
	m_currentVmdMolecule->add_rot(randomXRotation, 'x');
	m_currentVmdMolecule->add_rot(randomYRotation, 'y');
	m_currentVmdMolecule->add_rot(randomZRotation, 'z');

	setMoleculeAnimationRate();

	m_shouldAnimateMol = true;
}

void LindseyGame::displayMoreInfo() {
	m_labelUpdateTimer->stop();

	m_drawText = false;

	FileSpec spec;
	std::string fileName = m_moleculeFilesDiretoryPath;
	fileName.append("/").append(ms_moleculeFileSubdirectory);
	fileName.append("/").append(
			QString::number(m_currentMoleculeIndex).toLocal8Bit().data());
	fileName.append(ms_moleculeFileExtension);

	int currentVmdMoleculeId = ms_gameControllerInstance->m_vmdApp->molecule_load(-1,
			fileName.c_str(), NULL, &spec);
	m_currentVmdMolecule = ms_gameControllerInstance->m_vmdApp->moleculeList->mol_from_id(
			currentVmdMoleculeId);

	QPushButton* infoButton = m_window->findChild<QPushButton*>(
			QString(ms_moreInfoButtonName.c_str()));
	infoButton->hide();

	QPushButton* resumeButton = m_window->findChild<QPushButton*>(
			QString(ms_resumeButtonName.c_str()));
	resumeButton->hide();

	m_window->resize(m_window->width() + ms_infoAreaWidthAllowance,
			m_window->height());

	QWidget* infoArea = m_window->findChild<QWidget*>(
			QString(ms_infoAreaName.c_str()));

	MolData currentMoleculeData = m_moleculesList[m_currentMoleculeIndex];

	QLabel* molNameLabel = infoArea->findChild<QLabel*>(
			QString(ms_moleculeNameDisplayName.c_str()));
	std::string moleculeName = currentMoleculeData.molName;
	molNameLabel->setText(QString(moleculeName.c_str()));

	QLabel* molTypeLabel = infoArea->findChild<QLabel*>(
			QString(ms_moleculeTypeDisplayName.c_str()));
	std::string moleculeTypeName =
			m_moleculeChoiceTypes[currentMoleculeData.molType];
	molTypeLabel->setText("(" + QString(moleculeTypeName.c_str()) + ")");

	QLabel* molImageLabel = infoArea->findChild<QLabel*>(
			QString(ms_moleculeImageLabelName.c_str()));
	QPixmap moleculeImage(currentMoleculeData.molImageFile.c_str());
	molImageLabel->setPixmap(moleculeImage);

	WebLinkWidget* wikiLink = infoArea->findChild<WebLinkWidget*>(
			QString(ms_moleculeWebLinkLabelName.c_str()));
	wikiLink->setLinkAddress(currentMoleculeData.molWebPath);

	infoArea->show();
}

void LindseyGame::prepareToQuitOnlineGame() {
	///////////////////////////////
	// send quit request to boinc //
	//////////////////////////////

	// set up result section
	QFile defaultQuitRequestAddition(
			QString(ms_defaultQuitRequestFile.c_str()));
	QDomDocument requestAdditionDoc;
	requestAdditionDoc.setContent(&defaultQuitRequestAddition);

	// set the work unit name / package identifier
//    QDomElement packageIdentifierSection =
//        requestAdditionDoc.elementsByTagName (
//            QString (ms_workunitNameTag.c_str ())).at (0).toElement ();
//    packageIdentifierSection.firstChild ().setNodeValue (
//        QString (m_packageIdentifier.c_str ()));
	m_packageIdentifier.fillIntoQuitRequest(&requestAdditionDoc);

	// set the platform name aka selected category id
	QDomElement platformIdentifierSection =
			requestAdditionDoc.elementsByTagName(
					QString(ms_platformTag.c_str())).at(0).toElement();
	platformIdentifierSection.firstChild().setNodeValue(
			QString(m_selectedCategoryId.c_str()));

	QDomElement resultElement = requestAdditionDoc.documentElement();

	// set work_req_seconds to 0 to indicate nothing to do
	std::vector<ServerCommunicationManager::ReqData> changesToRequestBody = { {
			ms_quitRequestIndicatorTag, ms_quitRequestIndicator } };

	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();
	instance->makeBoicRequest(changesToRequestBody, &resultElement);

	///////////////////////////////////
	// delete files we've downloaded //
	///////////////////////////////////

	for (int i = 0; i < m_filesToDeleteOnExit.size(); ++i) {
		std::string filePath = ms_localFilesPath;
		filePath.append("/").append(m_filesToDeleteOnExit[i]);

		int result = remove(filePath.c_str());
		if (result != 0) {
			fprintf(stderr,
			"Error: Molecule Flashcards: Could not remove file %s\n",
			filePath.c_str ());
		}
	}
	m_filesToDeleteOnExit.clear();
}

void LindseyGame::prepareToQuitGame() {
	ErrorLog::logMessage("Preparing to quit Molecule Flashcards game");

	if (m_gameInOnlineMode) {
		prepareToQuitOnlineGame();
	}

	if (m_currentVmdMolecule != NULL) {
		ms_gameControllerInstance->m_vmdApp->molecule_delete(m_currentVmdMolecule->id());
	}
}

void LindseyGame::startNewGame() {
	prepareToQuitGame();

	ms_gameControllerInstance->m_vmdGlWindow->setParent(NULL);
	delete m_window;

	prepareToBeginGame();
}

void LindseyGame::alertServerError() {
	QMessageBox notification(m_window);
	notification.setText(
			"An error has been encountered while communicating with the server. This game cannot continue. Please select a different game or try again.");
	notification.setStandardButtons(QMessageBox::Ok);
	notification.setIcon(QMessageBox::Warning);
	notification.exec();

	// prevents error when window is deleted while starting new game
	notification.setParent(NULL);

	startNewGame();
}

}
