#ifndef SELECT_PAIR_HPP_
#define SELECT_PAIR_HPP_

#include <QtCore/QTime>

#include "Exscitech/Games/Game.hpp"
#include "Exscitech/Games/SelectionGame.hpp"

class ViewLigandGame;
class ViewProteinGame;
class MoleculeServerData;
namespace Exscitech
{
  class SelectProteinLigandPairGame : public Game, SelectionGameDelegate
  {

  public:

    SelectProteinLigandPairGame ();

    ~SelectProteinLigandPairGame ();

    void
    selectionFinished (void* selection);

    virtual void
    initWindow ();

    void
    update ();

    void
    handleKeyboardInput (int keyCode);

    void
    handleKeyboardUp (int key);

    bool
    handleMouseInput (int screenX, int screenY, int button);

    bool
    handleMouseMove (int screenX, int screenY);

    bool
    handleMouseRelease (int screenX, int screenY, int button);

    bool
    handleMouseWheel (int delta);

    bool
    handleWindowResize (int width, int height);

    virtual void
    drawGameGraphics ();

  private:

    enum GameState
    {
      INIT, PROTEIN, LIGAND
    };

    Game* m_currentSubGame;
    MoleculeServerData* m_selectedProtein;
    MoleculeServerData* m_selectedLigand;
    GameState m_state;
  };
}

#endif
