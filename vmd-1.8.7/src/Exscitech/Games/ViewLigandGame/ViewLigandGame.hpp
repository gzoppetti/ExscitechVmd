#ifndef VIEW_LIGAND_GAME_HPP_
#define VIEW_LIGAND_GAME_HPP_

#include <QtCore/QTime>

#include "Exscitech/Games/SelectionGame.hpp"

#include "Exscitech/Math/Vector2.hpp"

namespace Exscitech
{
  class MoleculeServerData;
  class Camera;
  class Drawable;
  class ViewLigandGame : public SelectionGame
  {

  public:

    ViewLigandGame (SelectionGameDelegate* delegate,
        MoleculeServerData* protein);

    ~ViewLigandGame ();

    void
    initWindow ();

    void
    onUpdate ();

    void
    handleKeyboardInput (int keyCode)
    {
    }

    void
    handleKeyboardUp (int key)
    {
    }

    bool
    handleMouseInput (int screenX, int screenY, int button);

    bool
    handleMouseMove (int screenX, int screenY);

    bool
    handleMouseRelease (int screenX, int screenY, int button)
    {
      return false;
    }

    bool
    handleMouseWheel (int delta);

    bool
    handleWindowResize (int width, int height)
    {
      return false;
    }

    void
    drawGameGraphics ();

  protected:

    virtual void
    selectionFinished (void* selection);

  private:

    void
    initList (QStringList& imageList, QStringList& textList);

    void*
    onSelectionFinished (int selection);

    void
    onSelectionChanged (int selection);

  private:

    Camera* m_camera;
    Drawable* m_molecule;

    std::vector<MoleculeServerData*> m_moleculeList;
    Vector2i m_mouseLocation;
    int m_lastUpdateTime;
  };
}

#endif
