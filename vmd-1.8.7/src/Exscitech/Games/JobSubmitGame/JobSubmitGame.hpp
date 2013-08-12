#ifndef JOBSUBMITGAME_HPP_
#define JOBSUBMITGAME_HPP_

#include <QtCore/QObject>

#include "Exscitech/Display/QtWindow.hpp"
#include "Exscitech/Games/Game.hpp"

#include "Exscitech/Math/Vector2.hpp"
#include "Exscitech/Math/Matrix3x3.hpp"

class QProgressDialog;
class QLabel;
class QPushButton;
class QDomDocument;

namespace Exscitech
{

  class ProteinServerData;
  class LigandServerData;
  class ConformationServerData;
  class ToggleWidget;
  class ImageSelectionWidget;
  class JobSubmitGame : public Game
  {

  Q_OBJECT

  public:

    JobSubmitGame ();

    virtual
    ~JobSubmitGame ();

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

  public Q_SLOTS:

    void
    primarySelectionChanged (int index);

    void
    secondarySelectionChanged (int index);

    void
    representationChanged (const QString & text);

    void
    displayToggleIndexChanged (int index);

    void
    rotationToggleIndexChanged (int index);

    void
    selectButtonPushed ();

    void
    backButtonPushed ();

  private:

    void
    createWindow ();

    void
    onProteinChanged (int index);

    void
    onLigandChanged (int index);

    void
    onConformationChanged(int index);

    void
    clearVmdMolecules ();

    void
    reloadMolecules ();

    Matrix3x3f
    calculateRelativeRotation ();

    void
    submitJob ();

    void
    beginProteinSelection ();

    void
    beginLigandSelection ();

    void
    beginOrientationSelection ();

    void
    extractMetaInfoFromResponse (QDomDocument* response);

    void
    extractProteinsFromResponse (QDomDocument* response);

    void
    extractConformationsFromResponse (QDomDocument* response);

    void
    extractVariablesFromResponse (QDomDocument* response);

    void
    extractInformationFromResponse (QDomDocument* response);

    void
    checkSubmissionStatus (QDomDocument* response);

    void
    downloadAllFiles ();

    void
    createThumbnails();

    void
    showNetworkErrorDialog ();

  private:

    static const int ms_borderSize;
    static const Vector2i ms_windowSize;
    static const Vector2i ms_glSize;
    static const Vector2i ms_primaryListSize;
    static const Vector2i ms_secondaryListSize;
    static const Vector2i ms_rotationTextSize;
    static const Vector2i ms_selectButtonSize;
    static const Vector2i ms_backButtonSize;
    static const Vector2i ms_webViewSize;
    static const Vector2i ms_repBoxSize;

    static const Vector2i ms_glPosition;
    static const Vector2i ms_primaryListPosition;
    static const Vector2i ms_secondaryListPosition;
    static const Vector2i ms_displayToggleWidgetPosition;
    static const Vector2i ms_rotationToggleWidgetPosition;
    static const Vector2i ms_rotationTextPosition;
    static const Vector2i ms_selectButtonPosition;
    static const Vector2i ms_backButtonPosition;
    static const Vector2i ms_webViewPosition;
    static const Vector2i ms_repBoxPosition;

    static const std::string ms_infoHeader;
    static const std::string ms_windowTitle;
    static const std::string ms_selectButtonText;
    static const std::string ms_submitButtonText;
    static const std::string ms_backButtonText;
    static const std::string ms_baseDownloadUrl;
    static const std::string ms_baseDownloadDirectory;

    static const std::string ms_serverReplyTag;
    static const std::string ms_sessionIdTag;
    static const std::string ms_nameTag;
    static const std::string ms_idTag;
    static const std::string ms_pdbUrlTag;

    static const std::string ms_proteinListTag;
    static const std::string ms_proteinTag;
    static const std::string ms_proteinDiseaseTag;
    static const std::string ms_proteinDescriptionTag;

    static const std::string ms_ligandListTag;
    static const std::string ms_ligandTag;

    static const std::string ms_conformationListTag;
    static const std::string ms_conformationTag;

    static const std::string ms_tempProfileTag;
    static const std::string ms_maxTag;
    static const std::string ms_minTag;
    static const std::string ms_maxTempTag;
    static const std::string ms_minTempTag;
    static const std::string ms_totalTimeTag;
    static const std::string ms_heatPercentTime;
    static const std::string ms_coolPercentTime;

    static const std::vector<std::string> ms_displayToggleButtons;
    static const std::vector<std::string> ms_rotationToggleButtons;

  private:

    enum GameState
    {
      SELECT_PROTEIN, SELECT_LIGAND, SELECT_ORIENTATION
    };

    enum DisplayToggles
    {
      DISPLAY_PROTEIN = 0, DISPLAY_MERGED = 1, DISPLAY_LIGAND = 2
    };

    enum RotationToggles
    {
      ROTATION_MERGED = 0, ROTATION_LIGAND = 1
    };

    struct JobVariableRanges
    {
      int maxTempMin;
      int maxtempMax;

      int minTempMin;
      int minTempMax;

      int totalTimeMin;
      int totalTimeMax;

      int heatTimeMin;
      int heatTimemax;

      int coolTimeMin;
      int coolTimeMax;
    };

  private:

    GameState m_state;
    QtWindow* m_window;
    ImageSelectionWidget* m_primaryListWidget;
    ImageSelectionWidget* m_secondaryListWidget;
    ToggleWidget* m_displayToggleWidget;
    ToggleWidget* m_rotationToggleWidget;
    QPushButton* m_backButton;
    QPushButton* m_submitButton;
    QLabel* m_rotationTextWidget;

    // Meta information from start game response
    std::string m_sessionId;
    std::string m_currentRepresentation;
    std::vector<ProteinServerData*> m_proteins;
    int m_selectedProteinIndex;
    int m_vmdProteinId;

    std::vector<LigandServerData*> m_ligands;
    int m_selectedConformationIndex;
    int m_selectedLigandIndex;
    int m_vmdConformationId;

    JobVariableRanges m_jobVariableRanges;
    int m_maxTemp;
    int m_minTemp;
    int m_totalTime;
    int m_heatPercent;
    int m_coolPercent;
  };
}

#endif
