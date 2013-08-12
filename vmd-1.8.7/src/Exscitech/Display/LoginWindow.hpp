#ifndef LOGINWINDOWUI_H_
#define LOGINWINDOWUI_H_

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QMainWindow>
#include <QtGui/QPushButton>
#include <QtGui/QStatusBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

namespace Exscitech
{
class GameController;
  class LoginWindow : public QWidget
  {
  Q_OBJECT

  public:
    LoginWindow (QWidget* parent = NULL);

  public slots:

    void
    login ();

    void
    playOffline ();

  protected:

    virtual void
    closeEvent (QCloseEvent* event);

  private:

    void
    setupUi ();

    void
    setupTopWidget();

    void
    setupCredentialWidget();

    void
    setupButtonWidget();

  private:
    static GameController* ms_gameControllerInstance;

    static const QString ms_welcomeText;
    static const QString ms_usernameHint;
    static const QString ms_passwordHint;
    static const QString ms_loginText;
    static const QString ms_offlineText;
    static const QString ms_loginErrorText;

    static const QString ms_errorLabelId;

  private:
    QWidget* m_centralwidget;
    QWidget* m_parentVerticalLayoutWidget;
    QVBoxLayout* m_parentVerticalLayout;
    QWidget* m_topWidget;
    QWidget* m_topVerticalLayoutWidget;
    QVBoxLayout* m_topVerticalLayout;
    QLabel* m_picture;
    QLabel* m_welcomeText;
    QWidget* m_credentialWidget;
    QWidget* m_credentialVerticalLayoutWidget;
    QVBoxLayout* m_credentialVerticalLayout;
    QLineEdit* m_usernameEdit;
    QLineEdit* m_passwordEdit;
    QWidget* m_buttonWidget;
    QWidget* m_buttonHorizontalLayoutWidget;
    QHBoxLayout* m_buttonHorizontalLayout;
    QPushButton* m_offlineButton;
    QPushButton* m_loginButton;
  };

}

#endif
