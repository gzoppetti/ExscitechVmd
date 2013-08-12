#ifndef GAME_CONTROLLER_HPP_
#define GAME_CONTROLLER_HPP_

#include <QtGui/QPushButton>
#include <string>

#include "Exscitech/Display/VmdGlWidget.hpp"

#include "Exscitech/Utilities/ErrorLog.hpp"

class VMDApp;

namespace Exscitech
{
  class QtVmdGlWindow;
  class QtWindow;
  class GameSelectionWindow;
  class Game;
  class QtOpenGLDisplayDevice;
  class LoginWindow;

  class GameController
  {
  public:

    enum ExscitechGame
    {
      MOLECULE_FLASHCARDS,
      IDENTIFICATION_GAME,
      JOB_SUBMIT_GAME,
      NUM_GAMES
    };

  public:

    static GameController* acquire();

    void
    initPlugin (VMDApp* vmdApp, QApplication* qtApp);

    void
    updatePlugin ();

    void
    shutdownPlugin ();

    bool
    shouldUpdate ();

    void
    handleKeyboardInput (int keyCode);

    void
    handleKeyboardUp (int key);

    bool
    handleMouseInput (int screenX, int screenY, Qt::MouseButton button);

    bool
    handleMouseMove (int screenX, int screenY);

    bool
    handleMouseRelease (int screenX, int screenY, Qt::MouseButton button);

    bool
    handleMouseWheel (int delta);

    bool
    handleWindowResize (int width, int height);

    void
    drawGraphics ();

    void
    stopCurrentGame ();

    void
    startNewGame (ExscitechGame game);

    void
    showLoginWindow ();

    void
    showGameSelectionWindow ();

    void
    terminateApplication ();

    void
    setOnlineMode (bool online);

    bool
    inOnlineMode ();

    bool
    isOnlineModeInvalid ();

    std::string
    getExscitechDirectory ();

    std::string
    createGameFolder (const std::string& folderName);

    void
    discardGameFolder (const std::string& folderName);

  private:

    GameController ();

  private:

    void
    initialize ();

    std::string
    createExscitechDirectories ();

  public:

    enum ControllerState
    {
      IDLE, GAME_CONSTRUCTED, INITIALIZE, UPDATE_LOOP
    };

  public:

    VMDApp* m_vmdApp;

    VmdGlWidget* m_vmdGlWindow;
    QApplication* m_qtApp;

    std::string m_username;
    std::string m_password;
    std::string m_applicationTitle;

    LoginWindow* m_loginWindow;

    bool m_vmdShouldAutoUpdate;

    std::string m_gameDataFolderName;
    std::string m_serverDataFolderName;

  private:

    bool m_inOnlineMode;
    bool m_onlineModeInvalid;

    ControllerState m_gameControllerState;

    GameSelectionWindow* m_gameChooserWindow;

    Game* m_currentGame;
  };

}

#endif
