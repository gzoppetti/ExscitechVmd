/*
 * This class provides information which must be supplied by all games for correct display in the game selection window
 * All new games should, after adding their ExscitechGame enum in GameConstroller, supply the appropriate information
 * by adding a case for their enum to each of the switch statements in each of these methods
 */

#ifndef GAME_INFO_MANAGER_HPP_
#define GAME_INFO_MANAGER_HPP_

#include <string>

#include "Exscitech/Games/GameController.hpp"

namespace Exscitech
{
  class GameInfoManager
  {
  public:

    enum GameType
    {
      LEARNING_GAME, JOB_SUBMIT_GAME, NO_TYPE_SPECIFIED
    };

  public:

    static GameType
    getGameType (GameController::ExscitechGame game);

    static std::string
    getGameTitle (GameController::ExscitechGame game);

    static std::string
    getGameIconPath (GameController::ExscitechGame game);

    static std::string
    getGameInstructionsPath (GameController::ExscitechGame game);

    static std::string
    getGameServerIdCode (GameController::ExscitechGame game);

    static bool
    gameHasOfflineMode (GameController::ExscitechGame game);

  private:

    GameInfoManager ();

  private:

    const static std::string ms_gameFilePath;
  };

}

#endif
