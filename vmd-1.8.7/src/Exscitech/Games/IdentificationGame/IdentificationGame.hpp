#ifndef IDENTIFICATION_GAME_HPP_
#define IDENTIFICATION_GAME_HPP_

#include <string>
#include <vector>

#include <QtGui/QBoxLayout>
#include <QtGui/QComboBox>

#include "Exscitech/Games/Game.hpp"
#include "Exscitech/Display/QtWindow.hpp"

class Molecule;

namespace Exscitech
{

  class IdentificationGame : public Game
  {
    // defined to use signals & slots
    Q_OBJECT

  public:

    IdentificationGame ();

    virtual
    ~IdentificationGame ();

    virtual void
    update ();

    virtual void
    initWindow ();

    virtual void
    handleKeyboardInput (int keyCode);

    virtual void
    handleKeyboardUp (int key);

    virtual bool
    handleMouseInput (int screenX, int screenY, int button);

    virtual bool
    handleMouseRelease (int screenX, int screenY, int button);

    virtual void
    drawGameGraphics ();

  public:
    // public members

    enum State
    {
      DROP_MOLECULE, PROCESS_CHOICE, INITIALIZING
    };

  public slots:

    void
    populateLevelList(int typeId);

    void
    displayFileChooser();

    void
    requestGameData();

    void
    readInGameData();

    void
    processChoice();

    void
    showInitialScreen();

  private:

    void
    constructTypeList(QComboBox* typeChooser);

    QHBoxLayout*
    createStartCancelButtons();

    void
    createInitialScreen();

    void
    constructGamePlayScreen(const std::vector<std::string>& choicesList);

    void
    constructGameScoreScreen();

    void
    initializeVmd();

    void
    startGame(const std::vector<std::string>& choicesList);

    bool
    directoryValid();

    bool
    readInSequence();

    bool
    readInChoices(std::vector<std::string>& choicesList);

    void
    setErrorText(std::string text);

    void
    processSkip();

    void
    loadNextInSequence();

    float
    calculateMoleculeDropRate();

    void
    dropMolecule();

    void
    handleEndOfGame();

  private:
    // private members

    QtWindow* m_gameWindow;
    QtWindow* m_initialWindow;

    std::string m_moleculeFilesDiretoryPath;

    std::vector<std::string> m_moleculeSequence;
    int m_currentMoleculeIndex;

    std::string m_moleculeSkippedString;

    std::string m_userChoiceSequence;

    State m_gameState;

    Molecule* m_currentMolecule;

    float m_moleculeDropRate;


  private:

    static int ms_initialWindowWidth;
    static int ms_initialWindowHeight;
    static int ms_finalWindowWidth;
    static int ms_finalWindowHeight;

    static int ms_quitButtonGapSize;

    static std::string ms_choicesListFileName;
    static std::string ms_sequenceListFileName;
    static std::string ms_moleculeFileExtension;

    static std::string ms_startGameText;
    static std::string ms_quitGameText;
    static std::string ms_newGameText;

    static std::string ms_typeLabelText;
    static std::string ms_levelLabelText;

    static std::string ms_offlineInstructions;

    static std::string ms_errorTextName;
    static std::string ms_errorMessage;

    static std::string ms_textBoxHeaderText;
    static std::string ms_lineEditName;
    static std::string ms_browseButtonText;

    static char ms_choicesListDelimiter;
    static char ms_sequenceListDelimiter;
    static char ms_userChoiceSequenceDelimiter;

    static std::string ms_offlineModeScoreLabel;
  };

}

#endif
