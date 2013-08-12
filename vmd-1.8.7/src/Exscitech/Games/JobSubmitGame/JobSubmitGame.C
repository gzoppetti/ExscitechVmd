#include <GL/glew.h>
#include <QtWebKit/QWebView>
#include <QtGui/QComboBox>
#include <QtGui/QMessageBox>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtXml/QDomDocument>
#include <QtGui/QProgressDialog>

#include <sstream>

#include "Exscitech/Games/JobSubmitGame/JobSubmitGame.hpp"
#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Display/ImageSelectionWidget.hpp"
#include "Exscitech/Display/VmdGlWidget.hpp"
#include "Exscitech/Display/ToggleWidget.hpp"

#include "Exscitech/Math/Matrix3x4.hpp"

#include "Exscitech/Utilities/TransformUtility.hpp"
#include "Exscitech/Utilities/ServerCommunicationManager.hpp"
#include "Exscitech/Utilities/ProteinServerData.hpp"
#include "Exscitech/Utilities/ConformationServerData.hpp"
#include "Exscitech/Utilities/LigandServerData.hpp"
#include "Exscitech/Utilities/ImageUtility.hpp"

#include "VMDApp.h"
#include "FPS.h"
#include "Axes.h"
#include "MoleculeList.h"

namespace Exscitech {
#define NONE_SELECTED -1

const int JobSubmitGame::ms_borderSize = 10;
const Vector2i JobSubmitGame::ms_windowSize(1024, 768);

const Vector2i JobSubmitGame::ms_primaryListSize(5.0 / 8 * ms_windowSize.x,
		120);
const Vector2i JobSubmitGame::ms_secondaryListSize(ms_primaryListSize.x - 100,
		80);
const Vector2i JobSubmitGame::ms_rotationTextSize(175, 32);
const Vector2i JobSubmitGame::ms_selectButtonSize(100, 48);
const Vector2i JobSubmitGame::ms_backButtonSize(100, 48);
const Vector2i JobSubmitGame::ms_glSize(5.0 / 8 * ms_windowSize.x,
		ms_windowSize.y - ms_selectButtonSize.y - ms_primaryListSize.y
				- 2 * ms_borderSize);
const Vector2i JobSubmitGame::ms_webViewSize(
		ms_windowSize.x - ms_glSize.x - ms_borderSize * 3,
		ms_windowSize.y - ms_borderSize * 2);
const Vector2i JobSubmitGame::ms_glPosition(ms_borderSize, ms_borderSize);
const Vector2i JobSubmitGame::ms_primaryListPosition(ms_borderSize,
		ms_glPosition.y + ms_glSize.y);
const Vector2i JobSubmitGame::ms_repBoxPosition(ms_borderSize, ms_borderSize);

// Relative to gl?
const Vector2i JobSubmitGame::ms_secondaryListPosition(
		ms_glSize.x / 2 - ms_secondaryListSize.x / 2,
		ms_glSize.y - ms_secondaryListSize.y);
const Vector2i JobSubmitGame::ms_displayToggleWidgetPosition(
		ms_glSize.x - 164 - ms_borderSize, ms_borderSize);
const Vector2i JobSubmitGame::ms_rotationToggleWidgetPosition(ms_borderSize,
		ms_glSize.y - 32);
const Vector2i JobSubmitGame::ms_rotationTextPosition(
		ms_glSize.x - ms_rotationTextSize.x,
		ms_displayToggleWidgetPosition.y + 32);
const Vector2i JobSubmitGame::ms_selectButtonPosition(
		ms_glPosition.x + ms_glSize.x / 2 - ms_selectButtonSize.x - 5,
		ms_primaryListPosition.y + ms_primaryListSize.y);
const Vector2i JobSubmitGame::ms_backButtonPosition(
		ms_glPosition.x + ms_glSize.x / 2 + 5,
		ms_primaryListPosition.y + ms_primaryListSize.y);
const Vector2i JobSubmitGame::ms_webViewPosition(
		ms_glPosition.x + ms_glSize.x + ms_borderSize, ms_borderSize);
const Vector2i JobSubmitGame::ms_repBoxSize(80, 32);

const std::string JobSubmitGame::ms_infoHeader = "info";
const std::string JobSubmitGame::ms_windowTitle = "Job Submit";
const std::string JobSubmitGame::ms_selectButtonText = "Select";
const std::string JobSubmitGame::ms_submitButtonText = "Submit";
const std::string JobSubmitGame::ms_backButtonText = "Back";
const std::string JobSubmitGame::ms_baseDownloadUrl =
		"docktest.gcl.cis.udel.edu/exscitech/";
const std::string JobSubmitGame::ms_baseDownloadDirectory =
		GameController::acquire()->getExscitechDirectory().append("/").append(
				GameController::acquire()->m_serverDataFolderName).append("/");
const std::string JobSubmitGame::ms_serverReplyTag = "server_reply";
const std::string JobSubmitGame::ms_sessionIdTag = "session_id";
const std::string JobSubmitGame::ms_nameTag = "name";
const std::string JobSubmitGame::ms_idTag = "id";
const std::string JobSubmitGame::ms_pdbUrlTag = "pdb_url";

const std::string JobSubmitGame::ms_proteinListTag = "protein_list";
const std::string JobSubmitGame::ms_proteinTag = "protein";
const std::string JobSubmitGame::ms_proteinDiseaseTag = "disease";
const std::string JobSubmitGame::ms_proteinDescriptionTag = "description";

const std::string JobSubmitGame::ms_ligandListTag = "ligand_list";
const std::string JobSubmitGame::ms_ligandTag = "ligand";

const std::string JobSubmitGame::ms_conformationListTag = "conformation_list";
const std::string JobSubmitGame::ms_conformationTag = "conformation";

const std::string JobSubmitGame::ms_tempProfileTag = "temp_profile_limits";
const std::string JobSubmitGame::ms_maxTag = "max";
const std::string JobSubmitGame::ms_minTag = "min";
const std::string JobSubmitGame::ms_maxTempTag = "max_temp";
const std::string JobSubmitGame::ms_minTempTag = "min_temp";
const std::string JobSubmitGame::ms_totalTimeTag = "total_time";
const std::string JobSubmitGame::ms_heatPercentTime = "heat_percent";
const std::string JobSubmitGame::ms_coolPercentTime = "cool_percent";

const std::vector<std::string> JobSubmitGame::ms_displayToggleButtons = {
		"Protein", "Merged", "Ligand" };
const std::vector<std::string> JobSubmitGame::ms_rotationToggleButtons = {
		"Merged", "Ligand" };

JobSubmitGame::JobSubmitGame() :
		m_state(SELECT_PROTEIN), m_selectedProteinIndex(NONE_SELECTED), m_vmdProteinId(
				NONE_SELECTED), m_selectedConformationIndex(NONE_SELECTED), m_selectedLigandIndex(
				NONE_SELECTED), m_vmdConformationId(NONE_SELECTED) {
	static GameController* instance = GameController::acquire();
	instance->m_vmdGlWindow->setVmdRespondMouse(true);
	instance->m_vmdGlWindow->setVmdRespondWheel(true);
	instance->m_vmdGlWindow->setVmdRespondKeys(true);
	instance->m_vmdApp->fps->off();
	instance->m_vmdApp->axes->off();
}

JobSubmitGame::~JobSubmitGame() {
	static GameController* instance = GameController::acquire();
	instance->m_vmdGlWindow->setParent(NULL);
	const QObjectList& children = instance->m_vmdGlWindow->children();
	for (int i = 0; i < children.size(); ++i) {
		delete children.at(i);
	}
	delete m_window;

	// Clear Molecules out of vmd scene
	instance->m_vmdApp->molecule_delete_all();
	instance->m_vmdGlWindow->restoreDefaults();
}

void JobSubmitGame::initWindow() {
	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();
	createWindow();
	QDomDocument* document = instance->makeStartJobSubmitRequest();
	extractInformationFromResponse(document);
	downloadAllFiles();
	createThumbnails();

	delete document;

	beginProteinSelection();
}

void JobSubmitGame::update() {
	if (m_state == SELECT_ORIENTATION) {
		Matrix3x3f relativeRotation = calculateRelativeRotation();

		float yaw, pitch, roll;
		relativeRotation.extractYawPitchRoll(yaw, pitch, roll);
		std::stringstream ss;
		ss << "<font color='white'>" << "X: " << std::fixed
				<< std::setprecision(2) << yaw << "  Y: " << std::fixed
				<< std::setprecision(2) << pitch << "  Z: " << std::fixed
				<< std::setprecision(2) << roll << "</font>";
		m_rotationTextWidget->setText(QString(ss.str().c_str()));
	}
}

void JobSubmitGame::handleKeyboardInput(int keyCode) {
}

void JobSubmitGame::handleKeyboardUp(int key) {

}

bool JobSubmitGame::handleMouseInput(int screenX, int screenY, int button) {
	return true;
}

bool JobSubmitGame::handleMouseRelease(int screenX, int screenY, int button) {
	return true;
}

void JobSubmitGame::drawGameGraphics() {
}

void JobSubmitGame::primarySelectionChanged(int index) {
	switch (m_state) {
	case SELECT_PROTEIN:
		onProteinChanged(index);
		break;

	case SELECT_LIGAND:
		onLigandChanged(index);
		break;

	case SELECT_ORIENTATION:
		break;
	}
}

void JobSubmitGame::secondarySelectionChanged(int index) {
	switch (m_state) {
	case SELECT_LIGAND:
		onConformationChanged(index);
		break;
	}
}

void JobSubmitGame::representationChanged(const QString & text) {
	static GameController* instance = GameController::acquire();
	m_currentRepresentation = text.toStdString();
	instance->m_vmdApp->moleculeList->set_default_representation(
			m_currentRepresentation.c_str());
	reloadMolecules();
}

void JobSubmitGame::displayToggleIndexChanged(int index) {
	static GameController* instance = GameController::acquire();
	switch (index) {
	case DISPLAY_PROTEIN: {
		if (m_vmdProteinId != NONE_SELECTED) {
			instance->m_vmdApp->molecule_display(m_vmdProteinId, 1);
		}
		if (m_vmdConformationId != NONE_SELECTED) {
			instance->m_vmdApp->molecule_display(m_vmdConformationId, 0);
		}
	}
		break;

	case DISPLAY_MERGED:
		if (m_vmdProteinId != NONE_SELECTED) {
			instance->m_vmdApp->molecule_display(m_vmdProteinId, 1);
		}
		if (m_vmdConformationId != NONE_SELECTED) {
			instance->m_vmdApp->molecule_display(m_vmdConformationId, 1);
		}
		break;

	case DISPLAY_LIGAND:
		if (m_vmdProteinId != NONE_SELECTED) {
			instance->m_vmdApp->molecule_display(m_vmdProteinId, 0);
		}
		if (m_vmdConformationId != NONE_SELECTED) {
			instance->m_vmdApp->molecule_display(m_vmdConformationId, 1);
		}
		break;
	}
}

void JobSubmitGame::rotationToggleIndexChanged(int index) {
	static GameController* instance = GameController::acquire();
	switch (index) {
	case ROTATION_MERGED: {
		instance->m_vmdApp->molecule_fix(m_vmdProteinId, 0);
		instance->m_vmdApp->molecule_fix(m_vmdConformationId, 0);
	}
		break;

	case ROTATION_LIGAND: {
		instance->m_vmdApp->molecule_fix(m_vmdProteinId, 1);
	}
		break;
	}
}

void JobSubmitGame::selectButtonPushed() {
	switch (m_state) {
	case SELECT_PROTEIN: {
		if (m_selectedProteinIndex != NONE_SELECTED) {
			beginLigandSelection();
		}
		break;
	}
	case SELECT_LIGAND: {
		if (m_selectedConformationIndex != NONE_SELECTED) {
			beginOrientationSelection();
		}
		break;
	}

	case SELECT_ORIENTATION: {
		submitJob();
	}
		break;
	}
}

void JobSubmitGame::backButtonPushed() {
	switch (m_state) {
	case SELECT_PROTEIN:
		break;

	case SELECT_LIGAND:
		beginProteinSelection();
		break;

	case SELECT_ORIENTATION:
		beginLigandSelection();
		break;
	}
}

void JobSubmitGame::onProteinChanged(int index) {
	m_selectedProteinIndex = index;
	fprintf(stderr, "protein selected: %d\n", m_selectedProteinIndex);

	if (m_displayToggleWidget->getSelectedIndex() == DISPLAY_LIGAND) {
		m_displayToggleWidget->setSelectedIndex(DISPLAY_MERGED);
	}
	reloadMolecules();
}

void JobSubmitGame::onLigandChanged(int index) {
	m_selectedLigandIndex = index;
	fprintf(stderr, "ligand selected: %d\n", m_selectedLigandIndex);

	if (m_displayToggleWidget->getSelectedIndex() == DISPLAY_PROTEIN) {
		m_displayToggleWidget->setSelectedIndex(DISPLAY_MERGED);
	}

	if (index != NONE_SELECTED) {
		m_secondaryListWidget->show();
	}

	m_secondaryListWidget->clear();
	for (ConformationServerData* data : m_ligands[m_selectedLigandIndex]->getConformations()) {
		m_secondaryListWidget->push_back(data->getThumbnailFilePath(),
				data->getConformationId());
	}
	m_selectedConformationIndex = 0;
	m_secondaryListWidget->selectIndex(0);
	reloadMolecules();
}

void JobSubmitGame::onConformationChanged(int index) {
	m_selectedConformationIndex = index;
	fprintf(stderr, "conformation selected: %d\n",
	m_selectedConformationIndex);

	if (m_displayToggleWidget->getSelectedIndex() == DISPLAY_PROTEIN) {
		m_displayToggleWidget->setSelectedIndex(DISPLAY_MERGED);
	}

	reloadMolecules();
}

void JobSubmitGame::clearVmdMolecules() {
	static GameController* instance = GameController::acquire();
	instance->m_vmdApp->molecule_delete_all();
	m_vmdConformationId = NONE_SELECTED;
	m_vmdProteinId = NONE_SELECTED;
}

/*
 * This method reloads the intended molecules into VMD.  This is useful for switching between phases or when representation changes.
 * As its name implies, all molecules intended to be shown have already been downloaded and are ready to load.
 */
void JobSubmitGame::reloadMolecules() {
	static GameController* instance = GameController::acquire();
	clearVmdMolecules();
	FileSpec spec;

	if (m_selectedProteinIndex > -1) {
		m_vmdProteinId = instance->m_vmdApp->molecule_load(-1,
				m_proteins[m_selectedProteinIndex]->getPdbFilePath().c_str(),
				NULL, &spec);

		if (m_displayToggleWidget->getSelectedIndex() == DISPLAY_LIGAND) {
			instance->m_vmdApp->molecule_display(m_vmdProteinId, 0);
		}
	}

	if (m_state != SELECT_PROTEIN && m_selectedLigandIndex > -1) {
		LigandServerData* ligand = m_ligands[m_selectedLigandIndex];
		ConformationServerData* conformation =
				ligand->getConformations()[m_selectedConformationIndex];
		m_vmdConformationId = instance->m_vmdApp->molecule_load(-1,
				conformation->getPdbFilePath().c_str(), NULL, &spec);

		if (m_displayToggleWidget->getSelectedIndex() == DISPLAY_PROTEIN) {
			instance->m_vmdApp->molecule_display(m_vmdConformationId, 0);
		}
	}

}

Matrix3x3f JobSubmitGame::calculateRelativeRotation() {
	static GameController* instance = GameController::acquire();
	Matrix3x4f proteinTransform = TransformUtility::getTransform(
			instance->m_vmdApp->moleculeList->mol_from_id(m_vmdProteinId));
	Matrix3x4f conformationTransform = TransformUtility::getTransform(
			instance->m_vmdApp->moleculeList->mol_from_id(m_vmdConformationId));

	Matrix3x3f proteinRotation = proteinTransform.rotation;
	Matrix3x3f conformationRotation = conformationTransform.rotation;

	Quaternion proteinQuaternion = proteinRotation.toQuaternion();
	Quaternion conformationQuaternion = conformationRotation.toQuaternion();

	conformationQuaternion.invert();
	Quaternion relativeRotationQuaternion = conformationQuaternion
			* proteinQuaternion;

	return relativeRotationQuaternion.toMatrix();
}

void JobSubmitGame::submitJob() {
	std::string sessionId = m_sessionId;
	std::string proteinId = m_proteins[m_selectedProteinIndex]->getId();
	std::string ligandId = m_ligands[m_selectedLigandIndex]->getId();
	std::string conformationId =
			m_ligands[m_selectedLigandIndex]->getConformations()[m_selectedConformationIndex]->getConformationId();
	Matrix3x3f relativeRotation = calculateRelativeRotation();
	Quaternion q = relativeRotation.toQuaternion();
	float rotationX = q.x;
	float rotationY = q.y;
	float rotationZ = q.z;
	float rotationPhi = q.w;
	int minTemp = m_minTemp;
	int maxTemp = m_maxTemp;
	int totalTime = m_totalTime;
	int heatPercent = m_heatPercent;
	int coolPercent = m_coolPercent;

	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();
	QDomDocument* response = instance->submitJob(sessionId, proteinId, ligandId,
			conformationId, rotationX, rotationY, rotationZ, rotationPhi,
			minTemp, maxTemp, totalTime, heatPercent, coolPercent);

	if (response != NULL) {
		checkSubmissionStatus(response);
	} else {
		showNetworkErrorDialog();
	}
}

void JobSubmitGame::createWindow() {
	static GameController* instance = GameController::acquire();
	m_window = new QtWindow();
	m_window->setFixedSize(ms_windowSize.x, ms_windowSize.y);
	instance->m_vmdGlWindow->setParent(m_window);
	instance->m_vmdGlWindow->setGeometry(ms_glPosition.x, ms_glPosition.y,
			ms_glSize.x, ms_glSize.y);

	instance->m_vmdGlWindow->show();

	QWebView* webView = new QWebView(m_window);
	webView->setUrl(QUrl("http://www.google.com"));
	webView->setGeometry(ms_webViewPosition.x, ms_webViewPosition.y,
			ms_webViewSize.x, ms_webViewSize.y);
	webView->show();

	QComboBox* repBox = new QComboBox(instance->m_vmdGlWindow);
	repBox->setGeometry(ms_repBoxPosition.x, ms_repBoxPosition.y,
			ms_repBoxSize.x, ms_repBoxSize.y);
	QStringList options;
	options.append(QString("Lines"));
	options.append(QString("CPK"));
	repBox->addItems(options);
	repBox->show();

	m_submitButton = new QPushButton(QString(ms_selectButtonText.c_str()),
			m_window);
	m_submitButton->setGeometry(ms_selectButtonPosition.x,
			ms_selectButtonPosition.y, ms_selectButtonSize.x,
			ms_selectButtonSize.y);
	m_submitButton->show();

	m_backButton = new QPushButton(QString(ms_backButtonText.c_str()),
			m_window);
	m_backButton->setGeometry(ms_backButtonPosition.x, ms_backButtonPosition.y,
			ms_backButtonSize.x, ms_backButtonSize.y);
	m_backButton->show();

	m_displayToggleWidget = new ToggleWidget(ms_displayToggleButtons, 0,
			instance->m_vmdGlWindow);
	m_displayToggleWidget->move(ms_displayToggleWidgetPosition.x,
			ms_displayToggleWidgetPosition.y);
	m_displayToggleWidget->show();

	m_rotationToggleWidget = new ToggleWidget(ms_rotationToggleButtons, 0,
			instance->m_vmdGlWindow);
	m_rotationToggleWidget->move(ms_rotationToggleWidgetPosition.x,
			ms_rotationToggleWidgetPosition.y);
	m_rotationToggleWidget->show();

	float secondaryImagePercent = (ms_secondaryListSize.y - 50.0)
			/ ms_secondaryListSize.y;
	m_secondaryListWidget = new ImageSelectionWidget(
			ms_secondaryListSize.y * secondaryImagePercent,
			ms_secondaryListSize.y * secondaryImagePercent,
			instance->m_vmdGlWindow);
	m_secondaryListWidget->setGeometry(ms_secondaryListPosition.x,
			ms_secondaryListPosition.y, ms_secondaryListSize.x,
			ms_secondaryListSize.y);
	m_secondaryListWidget->show();
	m_secondaryListWidget->setAutoFillBackground(true);

	float primaryListsize = (ms_primaryListSize.y - 50.0)
			/ ms_primaryListSize.y;
	m_primaryListWidget = new ImageSelectionWidget(
			ms_primaryListSize.y * primaryListsize,
			ms_primaryListSize.y * primaryListsize, m_window);
	m_primaryListWidget->setGeometry(ms_primaryListPosition.x,
			ms_primaryListPosition.y, ms_primaryListSize.x,
			ms_primaryListSize.y);
	m_primaryListWidget->show();

	m_rotationTextWidget = new QLabel(instance->m_vmdGlWindow);
	m_rotationTextWidget->setGeometry(ms_rotationTextPosition.x,
			ms_rotationTextPosition.y, ms_rotationTextSize.x,
			ms_rotationTextSize.y);
	m_rotationTextWidget->setAutoFillBackground(true);
	QPalette pal = m_rotationTextWidget->palette();
	pal.setColor(QPalette::Window, QColor(Qt::black));
	m_rotationTextWidget->setPalette(pal);
	m_rotationTextWidget->show();

	QObject::connect (m_primaryListWidget, SIGNAL(selectionChanged(int)), this,
			SLOT(primarySelectionChanged(int)));

	QObject::connect (m_secondaryListWidget, SIGNAL(selectionChanged(int)),
			this, SLOT(secondarySelectionChanged(int)));

	QObject::connect (repBox, SIGNAL(currentIndexChanged (const QString&)),
			this, SLOT(representationChanged(const QString&)));

	QObject::connect (m_displayToggleWidget, SIGNAL(itemSelected (int)), this,
			SLOT(displayToggleIndexChanged(int)));

	QObject::connect (m_rotationToggleWidget, SIGNAL(itemSelected (int)), this,
			SLOT(rotationToggleIndexChanged(int)));

	QObject::connect(m_submitButton, SIGNAL(clicked()), this,
			SLOT(selectButtonPushed()));

	QObject::connect(m_backButton, SIGNAL(clicked()), this,
			SLOT(backButtonPushed()));

	m_window->show();
}

/*
 * This method is used to set up the select protein phase.  Call it any time you want the game to be in the Select_Protein phase
 */
void JobSubmitGame::beginProteinSelection() {
	m_state = SELECT_PROTEIN;

	// Populate the primary list with proteins
	m_primaryListWidget->clear();
	for (unsigned int i = 0; i < m_proteins.size(); ++i) {
		m_primaryListWidget->push_back(m_proteins[i]->getThumbnailFilePath(),
				m_proteins[i]->getName());
	}

	if (m_selectedProteinIndex != NONE_SELECTED) {
		m_primaryListWidget->selectIndex(m_selectedProteinIndex);
	}

	// Change widgets
	m_secondaryListWidget->hide();
	m_rotationTextWidget->hide();
	m_primaryListWidget->show();
	m_rotationToggleWidget->hide();
	m_displayToggleWidget->hide();
	m_displayToggleWidget->setSelectedIndex(0);
	m_backButton->setEnabled(false);
}

/*
 * This method is used to set up the select ligand phase.  Call it any time you want the game to be in the Select_Ligand phase
 */
void JobSubmitGame::beginLigandSelection() {
	m_state = SELECT_LIGAND;
	m_primaryListWidget->clear();

	for (unsigned int i = 0; i < m_ligands.size(); ++i) {
		m_primaryListWidget->push_back(
				m_ligands[i]->getConformations()[0]->getThumbnailFilePath(),
				m_ligands[i]->getName());
	}

	if (m_selectedLigandIndex != NONE_SELECTED) {
		// We take a snapshot because changing the ligand selection resets the conformation index to 0.
		// This snapshot allows us to load the conformation the user had chosen before switching states
		int selectedConformationSnapshot = m_selectedConformationIndex;
		m_primaryListWidget->selectIndex(m_selectedLigandIndex);

		if (selectedConformationSnapshot != NONE_SELECTED) {
			m_secondaryListWidget->selectIndex(selectedConformationSnapshot);
		}
	}

	//Change Widgets

	m_rotationTextWidget->hide();
	m_submitButton->setText(QString(ms_selectButtonText.c_str()));
	m_primaryListWidget->show();
	m_rotationToggleWidget->hide();
	m_displayToggleWidget->show();
	m_displayToggleWidget->setSelectedIndex(1);
	m_backButton->setEnabled(true);
}

/*
 * This method is used to set up the select orientation phase.  Call it any time you want the game to be in the Select_Orientation phase
 */
void JobSubmitGame::beginOrientationSelection() {
	m_state = SELECT_ORIENTATION;

	m_secondaryListWidget->hide();
	m_rotationTextWidget->show();
	m_submitButton->setText(QString(ms_submitButtonText.c_str()));
	m_primaryListWidget->clear();
	m_primaryListWidget->hide();
	m_rotationToggleWidget->show();
	m_rotationToggleWidget->setSelectedIndex(0);
	//m_displayToggleWidget->hide ();
	m_displayToggleWidget->setSelectedIndex(1);
	m_backButton->setEnabled(true);
}

void JobSubmitGame::extractMetaInfoFromResponse(QDomDocument* response) {
	QDomElement mainElement = response->documentElement();
	QDomElement sessionIdElement = mainElement.firstChildElement(
			ms_sessionIdTag.c_str());

	m_sessionId = sessionIdElement.text().toStdString();
}

void JobSubmitGame::extractProteinsFromResponse(QDomDocument* response) {
	QDomElement mainElement = response->documentElement();
	QDomElement proteinListElement = mainElement.firstChildElement(
			ms_proteinListTag.c_str());
	QDomNodeList proteins = proteinListElement.elementsByTagName(
			ms_proteinTag.c_str());

	unsigned int numProteins = proteins.size();
	m_proteins.reserve(numProteins);
	for (unsigned int i = 0; i < numProteins; ++i) {
		QDomNode protein = proteins.at(i);
		QDomElement id = protein.firstChildElement(ms_idTag.c_str());
		QDomElement name = protein.firstChildElement(ms_nameTag.c_str());
		QDomElement disease = protein.firstChildElement(
				ms_proteinDiseaseTag.c_str());
		QDomElement pdbUrl = protein.firstChildElement(ms_pdbUrlTag.c_str());
		QDomElement description = protein.firstChildElement(
				ms_proteinDescriptionTag.c_str());

		ProteinServerData* data = new ProteinServerData();
		data->setId(id.text().toStdString());
		data->setName(name.text().toStdString());
		data->setDisease(disease.text().toStdString());
		data->setDownloadUrl(pdbUrl.text().toStdString());
		data->setNotes(description.text().toStdString());
		m_proteins.push_back(data);
	}
}

void JobSubmitGame::extractConformationsFromResponse(QDomDocument* response) {
	QDomElement mainElement = response->documentElement();
	QDomElement ligandListElement = mainElement.firstChildElement(
			ms_ligandListTag.c_str());
	QDomNodeList ligands = ligandListElement.elementsByTagName(
			ms_ligandTag.c_str());

	unsigned int numLigands = ligands.size();

	// We can assume at least one conformation per ligand, so might as well reserve some space
	m_ligands.reserve(numLigands);

	for (unsigned int i = 0; i < numLigands; ++i) {
		QDomNode ligand = ligands.at(i);
		QDomElement ligandId = ligand.firstChildElement(ms_idTag.c_str());
		QDomElement ligandName = ligand.firstChildElement(ms_nameTag.c_str());
		QDomElement conformationList = ligand.firstChildElement(
				ms_conformationListTag.c_str());
		QDomNodeList conformations = conformationList.elementsByTagName(
				ms_conformationTag.c_str());

		LigandServerData* data = new LigandServerData();
		data->setName(ligandName.text().toStdString());
		data->setId(ligandId.text().toStdString());

		std::vector<ConformationServerData*> conformationVector;

		unsigned int numConformations = conformations.size();
		for (unsigned int j = 0; j < numConformations; ++j) {
			QDomNode conformation = conformations.at(j);
			QDomElement conformationId = conformation.firstChildElement(
					ms_idTag.c_str());
			QDomElement conformationName = conformation.firstChildElement(
					ms_nameTag.c_str());
			QDomElement pdbUrl = conformation.firstChildElement(
					ms_pdbUrlTag.c_str());

			ConformationServerData* conformationData =
					new ConformationServerData();
			conformationData->setLigandId(ligandId.text().toStdString());
			conformationData->setConformationId(
					conformationId.text().toStdString());
			conformationData->setDownloadUrl(pdbUrl.text().toStdString());

			conformationVector.push_back(conformationData);
		}

		data->setConformations(conformationVector);
		m_ligands.push_back(data);
	}
}

void JobSubmitGame::extractVariablesFromResponse(QDomDocument* response) {
//      QDomElement mainElement = response->documentElement ();
//      QDomElement tempProfileLimits = mainElement.firstChildElement (
//          ms_tempProfileTag.c_str ());

}

void JobSubmitGame::extractInformationFromResponse(QDomDocument* response) {
	extractMetaInfoFromResponse(response);
	extractProteinsFromResponse(response);
	extractConformationsFromResponse(response);
	extractVariablesFromResponse(response);
}

void JobSubmitGame::checkSubmissionStatus(QDomDocument* response) {
	static GameController* instance = GameController::acquire();
	QDomElement mainElement = response->documentElement();
	QDomElement success = mainElement.firstChildElement("success");

	fprintf(stderr, "Text: %s", success.text ().toStdString ().c_str ());
	if (success.text() == QString("true")) {
		QMessageBox prompt;
		prompt.setText(
				"The job submission was successful.  You will be returned to the game selection window");
		prompt.setStandardButtons(QMessageBox::Ok);
		prompt.setDefaultButton(QMessageBox::Ok);
		prompt.setIcon(QMessageBox::Information);
		prompt.exec();
		instance->stopCurrentGame();
	} else {
		QMessageBox prompt;
		prompt.setText("The job submission was unsuccessful!  Uh oh?");
		prompt.setStandardButtons(QMessageBox::Ok);
		prompt.setDefaultButton(QMessageBox::Ok);
		prompt.setIcon(QMessageBox::Critical);
		prompt.exec();
		instance->stopCurrentGame();
	}
}

void JobSubmitGame::downloadAllFiles() {
	unsigned int numProteins = m_proteins.size();
	unsigned int numLigands = m_ligands.size();
	unsigned int numConformations = 0;
	for (unsigned int i = 0; i < numLigands; ++i) {
		numConformations += m_ligands[i]->getConformations().size();
	}
	unsigned int numFilesToDownload = numProteins + numConformations;
	unsigned int numFilesDownloaded = 0;

	QProgressDialog progress("Downloading files...", "Quit", 0,
			numFilesToDownload, m_window);
	progress.setWindowModality(Qt::WindowModal);
	progress.setValue(0);

	for (ProteinServerData* data : m_proteins) {
		std::stringstream labelText;

		labelText << "Downloading " << data->getName();
		progress.setLabelText(labelText.str().c_str());

		fprintf(stderr, "Downloading %s\n", labelText.str ().c_str ());
		std::string downloadUrl = ms_baseDownloadUrl;
		downloadUrl.append(data->getUrl());

		std::string filePath = ms_baseDownloadDirectory;
		std::stringstream ss;
		ss << "Protein-" << data->getName() << "-" << data->getId() << ".pdb";
		filePath.append(ss.str());

		static ServerCommunicationManager* instance =
				ServerCommunicationManager::acquire();
		instance->downloadFile(downloadUrl, filePath);
		data->setPdbFilePath(filePath);

		numFilesDownloaded++;
		progress.setValue(numFilesDownloaded);
	}

	for (LigandServerData* ligandData : m_ligands) {
		for (ConformationServerData* conformationData : ligandData->getConformations()) {
			std::stringstream labelText;
			labelText << "Downloading " << ligandData->getName() << " - "
					<< conformationData->getConformationId();

			fprintf(stderr, "Downloading %s\n", labelText.str ().c_str ());
			progress.setLabelText(labelText.str().c_str());
			std::string downloadUrl = ms_baseDownloadUrl;
			downloadUrl.append(conformationData->getUrl());

			std::string filePath = ms_baseDownloadDirectory;
			std::stringstream ss;
			ss << "Ligand-" << ligandData->getName() << ligandData->getId()
					<< "-" << "Conformation - "
					<< conformationData->getConformationId() << ".pdb";
			filePath.append(ss.str());

			static ServerCommunicationManager* instance =
					ServerCommunicationManager::acquire();
			instance->downloadFile(downloadUrl, filePath);
			conformationData->setPdbFilePath(filePath);

			numFilesDownloaded++;
			progress.setValue(numFilesDownloaded);
		}
	}
}

void JobSubmitGame::createThumbnails() {
	static GameController* instance = GameController::acquire();

	fprintf(stderr, "Creating Thumbnails\n");

	int numThumbnailsToCreate = 0;
	numThumbnailsToCreate += m_proteins.size();
	for (LigandServerData* data : m_ligands) {
		numThumbnailsToCreate += data->getConformations().size();
	}

	QProgressDialog progress("Creating Thumbnails...", "Quit", 0,
			numThumbnailsToCreate, m_window);
	progress.setWindowModality(Qt::WindowModal);
	progress.setValue(0);

	instance->m_vmdApp->moleculeList->set_default_representation("CPK");

	int numCreated = 0;
	for (ProteinServerData* data : m_proteins) {
		instance->m_vmdApp->molecule_delete_all();
		FileSpec spec;
		instance->m_vmdApp->molecule_load(-1, data->getPdbFilePath().c_str(),
				NULL, &spec);
		instance->m_vmdApp->VMDupdate(1);
		instance->m_vmdGlWindow->swapBuffers();
		std::stringstream ss;
		ss << ms_baseDownloadDirectory << "Protein-" << data->getName()
				<< ".jpg";
		fprintf(stderr, "saving...%s", ss.str ().c_str ());
		ImageUtility::saveFrameAsImage(ss.str());
		data->setThumbnailFilePath(ss.str());
		progress.setValue(++numCreated);
	}

	for (LigandServerData* data : m_ligands) {
		for (ConformationServerData* conformation : data->getConformations()) {
			instance->m_vmdApp->molecule_delete_all();
			FileSpec spec;
			instance->m_vmdApp->molecule_load(-1,
					conformation->getPdbFilePath().c_str(), NULL, &spec);

			instance->m_vmdApp->VMDupdate(1);
			instance->m_vmdGlWindow->swapBuffers();
			std::stringstream ss;
			ss << ms_baseDownloadDirectory << "Ligand -" << data->getId()
					<< "Conformation-" << conformation->getConformationId()
					<< ".jpg";
			fprintf(stderr, "saving...%s", ss.str ().c_str ());
			ImageUtility::saveFrameAsImage(ss.str());
			conformation->setThumbnailFilePath(ss.str());
			progress.setValue(++numCreated);
		}
	}

	instance->m_vmdApp->moleculeList->set_default_representation("Lines");
	instance->m_vmdApp->molecule_delete_all();
}
void JobSubmitGame::showNetworkErrorDialog() {
	QMessageBox prompt;
	prompt.setText(
			"Unable to connect to the ExSciTech server.  Please check your connection and try again");
	prompt.setStandardButtons(QMessageBox::Ok);
	prompt.setDefaultButton(QMessageBox::Ok);
	prompt.setIcon(QMessageBox::Critical);
	prompt.exec();
}
}
