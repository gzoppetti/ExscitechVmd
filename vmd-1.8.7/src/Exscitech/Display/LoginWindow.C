#include <QtGui/QCloseEvent>

#include <cstdio>

#include "Exscitech/Display/LoginWindow.hpp"
#include "Exscitech/Display/GameSelectionWindow.hpp"
#include "Exscitech/Games/GameController.hpp"

#include "Exscitech/Utilities/ServerCommunicationManager.hpp"

namespace Exscitech {

GameController* LoginWindow::ms_gameControllerInstance =
		GameController::acquire();
const QString LoginWindow::ms_welcomeText = "Welcome to Docking@Home with VMD!";
const QString LoginWindow::ms_usernameHint = "Email";
const QString LoginWindow::ms_passwordHint = "Password";
const QString LoginWindow::ms_loginText = "Login";
const QString LoginWindow::ms_offlineText = "Play Offline";
const QString LoginWindow::ms_loginErrorText =
		"<font color='red'>Your email or password is incorrect.</font>";

const QString LoginWindow::ms_errorLabelId = "ERROR_LABEL";

LoginWindow::LoginWindow(QWidget* parent) :
		QWidget(parent) {
	setupUi();
}

void LoginWindow::login() {
	QLabel* errorLabel = m_centralwidget->findChild<QLabel*>(ms_errorLabelId);
	errorLabel->setText("");

	std::string username = m_usernameEdit->text().toStdString();
	std::string password = m_passwordEdit->text().toStdString();

	static ServerCommunicationManager* instance =
			ServerCommunicationManager::acquire();
	instance->login(username, password);

	UserData currentUser = instance->getUserData();

	if (!currentUser.successful) {
		errorLabel->setText(ms_loginErrorText);
	} else {
		fprintf(stderr, "Login Window: Login Success!\n");
		//ServerCommunicationManager::requestJob ("JobReply.xml");

				m_passwordEdit->clear ();
				this->hide ();
				ms_gameControllerInstance->setOnlineMode (true);

				// GameSelectionWindow needs to be initialized after inOnlineMode set
				// if inOnlineMode changes later, window should be destroyed & recreated
				ms_gameControllerInstance->showGameSelectionWindow ();
			}
		}

void LoginWindow::playOffline() {
	m_passwordEdit->clear();
	this->hide();
	QLabel* errorLabel = m_centralwidget->findChild<QLabel*>(ms_errorLabelId);
	errorLabel->hide();
	ms_gameControllerInstance->setOnlineMode(false);
	ms_gameControllerInstance->showGameSelectionWindow();
}

void LoginWindow::closeEvent(QCloseEvent* event) {
	ms_gameControllerInstance->terminateApplication();
	event->accept();
}

void LoginWindow::setupUi() {
	this->resize(500, 350);
	this->setLayoutDirection(Qt::LeftToRight);
	m_centralwidget = new QWidget(this);
	m_parentVerticalLayoutWidget = new QWidget(m_centralwidget);
	m_parentVerticalLayoutWidget->setGeometry(QRect(0, 0, 500, 320));
	m_parentVerticalLayout = new QVBoxLayout(m_parentVerticalLayoutWidget);
	m_parentVerticalLayout->setSpacing(0);
	m_parentVerticalLayout->setContentsMargins(0, 0, 0, 0);

	setupTopWidget();
	setupCredentialWidget();
	setupButtonWidget();
}

void LoginWindow::setupTopWidget() {
	QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	sizePolicy.setHorizontalStretch(0);
	sizePolicy.setVerticalStretch(0);

	m_topWidget = new QWidget(m_parentVerticalLayoutWidget);
	sizePolicy.setHeightForWidth(m_topWidget->sizePolicy().hasHeightForWidth());
	m_topWidget->setSizePolicy(sizePolicy);
	m_topWidget->setMinimumSize(QSize(400, 175));
	m_topVerticalLayoutWidget = new QWidget(m_topWidget);
	m_topVerticalLayoutWidget->setGeometry(QRect(0, 0, 417, 154));
	m_topVerticalLayout = new QVBoxLayout(m_topVerticalLayoutWidget);
	m_topVerticalLayout->setSpacing(0);
	m_topVerticalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
	m_topVerticalLayout->setContentsMargins(0, 0, 0, 0);
	m_picture = new QLabel(m_topVerticalLayoutWidget);
	m_picture->setPixmap(
			QPixmap(
					QString::fromUtf8(
							"./vmd-1.8.7/ExscitechResources/DockingLoginPicture.png")));
	m_topVerticalLayout->addWidget(m_picture, 0,
			Qt::AlignHCenter | Qt::AlignTop);
	m_welcomeText = new QLabel(m_topVerticalLayoutWidget);
	m_welcomeText->setText(ms_welcomeText);
	m_topVerticalLayout->addWidget(m_welcomeText, 0, Qt::AlignHCenter);

	m_parentVerticalLayout->addWidget(m_topWidget, 0,
			Qt::AlignHCenter | Qt::AlignTop);
}

void LoginWindow::setupCredentialWidget() {
	QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	sizePolicy.setHorizontalStretch(0);
	sizePolicy.setVerticalStretch(0);

	m_credentialWidget = new QWidget(m_parentVerticalLayoutWidget);
	m_credentialVerticalLayoutWidget = new QWidget(m_credentialWidget);
	m_credentialVerticalLayoutWidget->setGeometry(QRect(50, 0, 401, 70));
	m_credentialVerticalLayout = new QVBoxLayout(
			m_credentialVerticalLayoutWidget);
	m_credentialVerticalLayout->setContentsMargins(0, 0, 0, 0);
	m_usernameEdit = new QLineEdit(m_credentialVerticalLayoutWidget);
	sizePolicy.setHeightForWidth(
			m_usernameEdit->sizePolicy().hasHeightForWidth());
	m_usernameEdit->setSizePolicy(sizePolicy);
	m_usernameEdit->setMinimumSize(QSize(250, 0));
	// TODO: REMOVE!!
	m_usernameEdit->setText("etkimmel@cs.millersville.edu");
	m_usernameEdit->setPlaceholderText(ms_usernameHint);
	m_credentialVerticalLayout->addWidget(m_usernameEdit, 0, Qt::AlignHCenter);

	m_passwordEdit = new QLineEdit(m_credentialVerticalLayoutWidget);
	sizePolicy.setHeightForWidth(
			m_passwordEdit->sizePolicy().hasHeightForWidth());
	m_passwordEdit->setSizePolicy(sizePolicy);
	m_passwordEdit->setMinimumSize(QSize(250, 0));
	m_passwordEdit->setEchoMode(QLineEdit::Password);
	// TODO: REMOVE!!
	m_passwordEdit->setText("tech2010");
	m_passwordEdit->setPlaceholderText(ms_passwordHint);
	m_credentialVerticalLayout->addWidget(m_passwordEdit, 0, Qt::AlignHCenter);

	QLabel* errorMessageLabel = new QLabel(m_credentialVerticalLayoutWidget);
	errorMessageLabel->setFixedSize(QSize(250, 30));
	errorMessageLabel->setWordWrap(true);
	errorMessageLabel->setAlignment(Qt::AlignCenter);
	errorMessageLabel->setObjectName(ms_errorLabelId);
	m_credentialVerticalLayout->addWidget(errorMessageLabel, 0,
			Qt::AlignHCenter);

	m_parentVerticalLayout->addWidget(m_credentialWidget);
}

void LoginWindow::setupButtonWidget() {
	QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	sizePolicy.setHorizontalStretch(0);
	sizePolicy.setVerticalStretch(0);

	m_buttonWidget = new QWidget(m_parentVerticalLayoutWidget);
	m_buttonHorizontalLayoutWidget = new QWidget(m_buttonWidget);
	m_buttonHorizontalLayoutWidget->setGeometry(QRect(150, 0, 200, 50));
	m_buttonHorizontalLayout = new QHBoxLayout(m_buttonHorizontalLayoutWidget);
	m_buttonHorizontalLayout->setContentsMargins(0, 0, 0, 0);
	m_offlineButton = new QPushButton(m_buttonHorizontalLayoutWidget);
	m_offlineButton->setText(ms_offlineText);
	m_offlineButton->sizePolicy().hasHeightForWidth();
	m_offlineButton->setSizePolicy(sizePolicy);
	m_buttonHorizontalLayout->addWidget(m_offlineButton, 0, Qt::AlignHCenter);

	m_loginButton = new QPushButton(m_buttonHorizontalLayoutWidget);
	m_loginButton->setText(ms_loginText);
	m_loginButton->sizePolicy().hasHeightForWidth();
	m_loginButton->setSizePolicy(sizePolicy);

	m_buttonHorizontalLayout->addWidget(m_loginButton, 0, Qt::AlignHCenter);
	m_parentVerticalLayout->addWidget(m_buttonWidget);

	QObject::connect(m_offlineButton, SIGNAL(clicked()), this,
			SLOT(playOffline()));
	QObject::connect(m_loginButton, SIGNAL(clicked()), this, SLOT(login()));

}
}
