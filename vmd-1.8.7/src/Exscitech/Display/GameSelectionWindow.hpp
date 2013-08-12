#ifndef MU_QTPICKGAME_WINDOW_HPP_
#define MU_QTPICKGAME_WINDOW_HPP_

#include <QtGui/QWidget>
#include <QtCore/QObject>
#include <QtGui/QLabel>
#include <QtWebKit/QWebView>

#include "Exscitech/Display/GameChoiceWidget.hpp"

#include <string>

namespace Exscitech
{
  class GameSelectionWindow : public QWidget
  {
  Q_OBJECT

  public:

    GameSelectionWindow (QWidget* parent = NULL);

  public slots:

    void
    handleGameSelection();

    void
    playSelectedGame();

    void
    logoutAndReturn();

  protected:

     virtual void
     closeEvent (QCloseEvent* event);

  private:

     QWidget*
     constructGameSelectionTab();

     QLayout*
     createGameDisplayArea();

     QLayout*
     createGameInfoArea();

     QWidget*
     constructStatisticsTab();

  private:

     static GameController* ms_gameControllerInstance;

     static std::string ms_gamesTabTitle;
     static std::string ms_statsTabTitle;

     static std::string ms_offlineModeGreeting;

     static std::string ms_learningGamesTitle;
     static std::string ms_dockingGamesTitle;

     static std::string ms_defaultGameInfoTitle;
     static std::string ms_defaultGameInfoText;

     static std::string ms_noGameInstructionsText;

     static std::string ms_onlineLogoutText;
     static std::string ms_offlineLogoutText;

     static int ms_gameListsGapSize;

  private:

     QLabel* m_gameInfoTitle;
     QWebView* m_gameInfoInstructionDisplay;
     QPushButton* m_gamePlayButton;

     GameChoiceWidget* m_selectedGameWidget;

  };
}

#endif
