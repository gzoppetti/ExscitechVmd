#ifndef LINDSEY_GAME_HPP_
#define LINDSEY_GAME_HPP_

#include <string>
#include <vector>
#include <QtGui/QWidget>
#include <QtGui/QComboBox>
#include <QtCore/QTimer>

#include "Exscitech/Games/Game.hpp"
#include "Exscitech/Utilities/WorkunitId.hpp"
#include "Exscitech/Display/QtWindow.hpp"

class VMDApp;
class Molecule;

namespace Exscitech
{

  struct MolData
  {
    std::string molName;
    int molType;
    std::string molImageFile;
    std::string molWebPath;
  };

  class GameController;
  class LindseyGame : public Game
  {
    // defined to use signals & slots
  Q_OBJECT

  public:

    LindseyGame ();

    virtual
    ~LindseyGame ();

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

  public slots:
    // button callback

    void
    displayFileChooser ();

    void
    initOnlineGame ();

    void
    readGameData ();

    void
    processChoice ();

    void
    resumeGame ();

    void
    pauseOnlineGame ();

    void
    updateLabelText ();

    void
    displayMoreInfo ();

    void
    startNewGame ();

  private:

    void
    prepareToBeginGame ();

    void
    showInitialWindow ();

    QComboBox*
    createOnlinePackageList ();

    void
    createGameWindow ();

    QWidget*
    createInfoArea ();

    bool
    initMoleculeLists ();

    bool
    getPackageFromServer ();

    void
    startGame ();

    void
    setErrorText (const std::string& text);

    void
    displayCategoryLoadingError ();

    void
    animateMolecule ();

    void
    loadNewMolecule ();

    void
    processSkip ();

    void
    sendChoiceToServer (int categoryChoice);

    void
    updateWinRate ();

    void
    updateButtons ();

    void
    updateSpeedLevel (bool wasCorrect);

    void
    setMoleculeAnimationRate ();

    void
    prepareToQuitGame ();

    void
    prepareToQuitOnlineGame ();

    void
    alertServerError ();

  private:
    // private members
    bool m_gameInOnlineMode;

    QtWindow* m_initialWindow;
    QtWindow* m_window;

    std::string m_moleculeFilesDiretoryPath;

    std::vector<std::string> m_moleculeChoiceTypes;
    std::vector<MolData> m_moleculesList;

    int m_currentGameLevel;
    int m_numRight;
    int m_numWrong;

    // this holds the mol id of mol to show in offline mode
    int m_currentMoleculeIndex;
    // this holds the file name of mol to show in online mode
    std::string m_currentMoleculeFile;

    Molecule* m_currentVmdMolecule;

    bool m_shouldAnimateMol;
    bool m_isVerticalDrop;

    float m_molAnimationRates[2];

    bool m_drawText;
    std::string m_textToDraw;

    QTimer* m_labelUpdateTimer;

    // used only in online mode
    WorkunitId m_packageIdentifier;
    std::string m_selectedCategoryId;
    std::vector<std::string> m_filesToDeleteOnExit;

  private:

    static GameController* ms_gameControllerInstance;

    const static std::string ms_selectLabelText;
    const static int ms_initialWindowMiddleSpace;
    const static std::string ms_chooseOwnLabelText[];
    const static std::string ms_errorTextName;
    const static std::string ms_textBoxHeaderText;
    const static std::string ms_lineEditName;
    const static std::string ms_browseButtonText;
    const static std::string ms_startGameText;
    const static std::string ms_startOfflineGameText;
    const static std::string ms_quitGameText;
    const static std::string ms_errorMessage;
    const static std::string ms_moleculeDataFileName;
    const static char ms_moleculeDataDelimiter;
    const static char ms_moleculeDataChoiceDelimiter;
    const static std::string ms_moleculeFileSubdirectory;
    const static std::string ms_moleculeFileExtension;
    const static std::string ms_imageFileSubdirectory;
    const static int ms_numImageFileExtensions;
    const static std::string ms_imageFileExtensions[2];
    const static int ms_initialWindowMinWidth;
    const static int ms_initialWindowMinHeight;
    const static int ms_windowMinWidth;
    const static int ms_windowMinHeight;
    const static int ms_quitButtonGapSize;
    const static std::string ms_offlineModeTitle;
    const static std::string ms_correctMessage;
    const static std::string ms_incorrectMessage;
    const static std::string ms_skippedMessage;
    const static std::string ms_winRateLabelName;
    const static std::string ms_gameLevelLabelName;
    const static std::string ms_scoreLabelName;
    const static std::string ms_winRateLabelText;
    const static std::string ms_gameLevelLabelText;
    const static std::string ms_scoreLabelText;

    const static std::string ms_moreInfoButtonText;
    const static std::string ms_moreInfoButtonName;
    const static std::string ms_resumeButtonTextPart1;
    const static std::string ms_resumeButtonTextPart2;
    const static std::string ms_resumeButtonTextShort;
    const static std::string ms_pauseButtonText;
    const static std::string ms_resumeButtonName;
    const static std::string ms_pauseButtonName;

    const static int ms_pauseBetweenInMs;
    const static int ms_updateLabelInMs;

    const static int ms_infoWidth;
    const static int ms_infoheight;
    const static int ms_infoOffsetFromBottom;
    const static int ms_resumeWidth;
    const static int ms_resumeHeight;
    const static int ms_resumeOffsetFromBottom;

    const static std::string ms_infoAreaName;
    const static int ms_infoAreaTopSpace;
    const static int ms_infoAreaMidSpace;
    const static int ms_infoAreaWidthAllowance;

    const static std::string ms_moleculeNameDisplayName;
    const static std::string ms_moleculeTypeDisplayName;
    const static std::string ms_moleculeImageLabelName;
    const static std::string ms_moleculeWebLinkLabelName;

    const static std::string ms_packageChooserName;

    const static std::string ms_moleculeWikiLinkTitle;

    /* For Server: tags used */
    const static std::string ms_gameIdTag;
    const static std::string ms_categoryTag;
    const static std::string ms_categoryIdAttr;
    const static std::string ms_boincCategoryIdTag;
    // this for identifying work unit in Game Specific reply/request
    const static std::string ms_workunitSpecificNameTag;
    const static std::string ms_platformTag;
    const static std::string ms_moleculeIdentifierTag;
    const static std::string ms_userCategoryChoiceTag;
    const static std::string ms_wasResponseCorrectTag;
    const static std::string ms_moleculeIdentifierAttr;
    const static std::string ms_choiceCorrectAffirmative;
    const static std::string ms_scoreTag;
    const static std::string ms_nextMoleculeTag;
    const static std::string ms_textDisplayTag;
    const static std::string ms_imageTag;
    const static std::string ms_webAddressTag;
    const static std::string ms_requestIdTag;

    const static std::string ms_categoryRequestId;
    const static std::string ms_nextMolRequestId;

    const static std::string ms_endOfGameNotice;
    const static std::string ms_quitRequestIndicator;
    const static std::string ms_quitRequestIndicatorTag;

    const static std::string ms_defaultQuitRequestFile;

    const static std::string ms_optionsFileName;
    const static std::string ms_localFilesPath;

    const static std::string ms_gameEndMessage;

  };

}

#endif
