#ifndef VIEW_PROTEIN_GAME_HPP_
#define VIEW_PROTEIN_GAME_HPP_

#include <QtCore/QTime>

#include "Exscitech/Utilities/MoleculeServerData.hpp"
#include "Exscitech/Games/SelectionGame.hpp"
#include "Exscitech/Display/Camera.hpp"
#include "Exscitech/Math/Vector2.hpp"

namespace Exscitech
{
  class SSAO;
  class FullQuad;

  class ViewProteinGame : public SelectionGame
  {

  public:

    ViewProteinGame (SelectionGameDelegate* delegate);

    ~ViewProteinGame ();

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

  private:

    void
    initList (QStringList& imageList, QStringList& textList);

    void*
    onSelectionFinished (int selection);

    void
    onSelectionChanged (int selection);

  private:

    SSAO* m_ssao;
    Camera* m_camera;
    Drawable* m_molecule;

    std::vector<MoleculeServerData*> m_moleculeList;
    Vector2i m_mouseLocation;
    int m_lastUpdateTime;
  };
}

#endif
