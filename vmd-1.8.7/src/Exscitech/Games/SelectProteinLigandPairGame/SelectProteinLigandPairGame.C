#include "Exscitech/Games/ViewProteinGame/ViewProteinGame.hpp"
#include "Exscitech/Games/ViewLigandGame/ViewLigandGame.hpp"
#include "Exscitech/Utilities/MoleculeServerData.hpp"

#include "Exscitech/Games/SelectProteinLigandPairGame/SelectProteinLigandPairGame.hpp"

namespace Exscitech
{
  SelectProteinLigandPairGame::SelectProteinLigandPairGame () :
      m_currentSubGame (NULL), m_selectedProtein (NULL), m_selectedLigand (
          NULL), m_state (INIT)
  {
  }

  SelectProteinLigandPairGame::~SelectProteinLigandPairGame ()
  {
    delete m_currentSubGame;
    delete m_selectedProtein;
    delete m_selectedLigand;
  }

  void
  SelectProteinLigandPairGame::selectionFinished (void* selection)
  {
    switch (m_state)
    {
      case PROTEIN:
        m_selectedProtein = new MoleculeServerData (
            reinterpret_cast<MoleculeServerData*> (selection));

        delete m_currentSubGame;
        m_currentSubGame = NULL;
        m_currentSubGame = new ViewLigandGame (this, m_selectedProtein);
        m_state = LIGAND;
        break;

      case LIGAND:
        m_selectedLigand = new MoleculeServerData (
                    reinterpret_cast<MoleculeServerData*> (selection));

        // Send results to server....

        // TODO: Confirmation of sent pair.
        GameController::stopCurrentGame ();
        break;

      default:
        break;
    }
  }

  void
  SelectProteinLigandPairGame::initWindow ()
  {

  }

  void
  SelectProteinLigandPairGame::update ()
  {
    switch (m_state)
    {
      case INIT:
      {
        m_currentSubGame = new ViewProteinGame (this);
        m_state = PROTEIN;
      }
        break;
      case PROTEIN:
      case LIGAND:
        m_currentSubGame->update ();
        break;
    }
  }

  void
  SelectProteinLigandPairGame::handleKeyboardInput (int keyCode)
  {
    if (m_currentSubGame != NULL)
    {
      m_currentSubGame->handleKeyboardInput (keyCode);
    }
  }

  void
  SelectProteinLigandPairGame::handleKeyboardUp (int key)
  {
    if (m_currentSubGame != NULL)
    {
      m_currentSubGame->handleKeyboardUp (key);
    }
  }

  bool
  SelectProteinLigandPairGame::handleMouseInput (int screenX, int screenY,
      int button)
  {
    if (m_currentSubGame != NULL)
    {
      return m_currentSubGame->handleMouseInput (screenX, screenY, button);
    }
    return false;
  }

  bool
  SelectProteinLigandPairGame::handleMouseMove (int screenX, int screenY)
  {
    if (m_currentSubGame != NULL)
    {
      return m_currentSubGame->handleMouseMove (screenX, screenY);
    }
    return false;
  }

  bool
  SelectProteinLigandPairGame::handleMouseRelease (int screenX, int screenY,
      int button)
  {
    if (m_currentSubGame != NULL)
    {
      return m_currentSubGame->handleMouseRelease (screenX, screenY, button);
    }
    return false;
  }

  bool
  SelectProteinLigandPairGame::handleMouseWheel (int delta)
  {
    if (m_currentSubGame != NULL)
    {
      return m_currentSubGame->handleMouseWheel (delta);
    }
    return false;
  }

  bool
  SelectProteinLigandPairGame::handleWindowResize (int width, int height)
  {
    if (m_currentSubGame != NULL)
    {
      return m_currentSubGame->handleWindowResize (width, height);
    }
    return false;
  }

  void
  SelectProteinLigandPairGame::drawGameGraphics ()
  {
    if (m_currentSubGame != NULL)
    {
      m_currentSubGame->drawGameGraphics ();
    }
  }
}
