#include "Exscitech/Games/GameInfoManager.hpp"

namespace Exscitech
{
  const std::string GameInfoManager::ms_gameFilePath =
      "vmd-1.8.7/src/Exscitech/Games/";

  GameInfoManager::GameType
  GameInfoManager::getGameType (GameController::ExscitechGame game)
  {
    GameType gameType = NO_TYPE_SPECIFIED;
    switch (game)
    {

      case GameController::MOLECULE_FLASHCARDS:
      case GameController::IDENTIFICATION_GAME:
        gameType = LEARNING_GAME;
        break;

      case GameController::JOB_SUBMIT_GAME:
        gameType = JOB_SUBMIT_GAME;
        break;

      default:
        break;
    }
    return (gameType);
  }

  std::string
  GameInfoManager::getGameTitle (GameController::ExscitechGame game)
  {
    std::string gameTitle = "No Title Provided";
    switch (game)
    {

      case GameController::MOLECULE_FLASHCARDS:
        gameTitle = "Molecule Flashcards";
        break;

      case GameController::IDENTIFICATION_GAME:
        gameTitle = "Identification Game";
        break;

      case GameController::JOB_SUBMIT_GAME:
        gameTitle = "Docking Submission";
        break;

      default:
        break;
    }
    return (gameTitle);
  }

  std::string
  GameInfoManager::getGameIconPath (GameController::ExscitechGame game)
  {
    std::string gameIconPath = "";
    switch (game)
    {

      case GameController::MOLECULE_FLASHCARDS:
        gameIconPath.append (ms_gameFilePath).append (
            "LindseyGame/IdentificationIcon.jpg");
        break;

      case GameController::IDENTIFICATION_GAME:
        gameIconPath.append (ms_gameFilePath).append (
            "IdentificationGame/IdIcon.png");
        break;

      case GameController::JOB_SUBMIT_GAME:
        break;

      default:
        break;
    }
    return (gameIconPath);
  }

  std::string
  GameInfoManager::getGameInstructionsPath (GameController::ExscitechGame game)
  {
    std::string gameInstructionsPath = "";
    switch (game)
    {

      case GameController::MOLECULE_FLASHCARDS:
        gameInstructionsPath.append (ms_gameFilePath).append (
            "LindseyGame/Instructions.html");
        break;

      case GameController::IDENTIFICATION_GAME:
        gameInstructionsPath.append (ms_gameFilePath).append (
            "IdentificationGame/IdGameInstructions.html");
        break;

      case GameController::JOB_SUBMIT_GAME:
        break;

      default:
        break;
    }
    return (gameInstructionsPath);
  }

  // Game id recognized by server for each game
  // games that communicate with the server should have this set
  std::string
  GameInfoManager::getGameServerIdCode (GameController::ExscitechGame game)
  {
    std::string gameIdCode = "";
    switch (game)
    {
      case GameController::MOLECULE_FLASHCARDS:
        gameIdCode = "LG";
        break;

      default:
        break;
    }
    return (gameIdCode);
  }

  bool
  GameInfoManager::gameHasOfflineMode (GameController::ExscitechGame game)
  {
    bool hasOfflineMode = false;
    switch (game)
    {

      case GameController::MOLECULE_FLASHCARDS:
      case GameController::IDENTIFICATION_GAME:
        hasOfflineMode = true;
        break;

      case GameController::JOB_SUBMIT_GAME:
        hasOfflineMode = false;
        break;

      default:
        break;
    }
    return (hasOfflineMode);
  }

}
