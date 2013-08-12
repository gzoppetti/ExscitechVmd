#include <QtGui/QFont>
#include <QtGui/QStyle>
#include <QtGui/QBoxLayout>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QListWidget>

#include <GL/glew.h>

#include "VMDApp.h"
#include "Axes.h"
#include "Scene.h"

#include "Exscitech/Display/QtWindow.hpp"

#include "Exscitech/Games/GameInfoManager.hpp"
#include "Exscitech/Games/GameController.hpp"

#include "Exscitech/Games/ViewLigandGame/ViewLigandGame.hpp"

#include "Exscitech/Graphics/LabeledMolecule.hpp"
#include "Exscitech/Graphics/BallAndStickMolecule.hpp"

#include "Exscitech/Graphics/MoleculeLoader.hpp"

#include "Exscitech/Constants.hpp"

#include "Exscitech/Utilities/MoleculeServerData.hpp"

namespace Exscitech
{

  ViewLigandGame::ViewLigandGame (SelectionGameDelegate* delegate,
      MoleculeServerData* protein) :
      SelectionGame (delegate), m_molecule (NULL)
  {
    m_camera = new Camera (Vector4i (0, 0, ms_glSize.x, ms_glSize.y), 60.f,
        0.01f, 1000.f);
    m_camera->moveBackward (50);

    m_menuWindow->setWindowTitle (QString ("View Ligand"));
  }

  ViewLigandGame::~ViewLigandGame ()
  {
    delete m_camera;
    delete m_molecule;

    for (MoleculeServerData* data : m_moleculeList)
    {
      delete data;
    }

    m_moleculeList.clear ();
  }

  void
  ViewLigandGame::initList (QStringList& imageList, QStringList& textList)
  {
    // TODO: Receive list from server.
    m_moleculeList.reserve (100);

    for (int i = 0; i < 10; ++i)
    {
      std::stringstream ss;
      ss << "Ligand #" << i;
      std::string label = ss.str ();

      MoleculeServerData* data = new MoleculeServerData ();
      data->setDownloadUrl ("TODO")->setId ("Id")->setMoleculeName ("Ligand!")->setNotes (
          "Notes!")->setPdbFilePath (
          "./vmd-1.8.7/ExscitechResources/ligand.pdb");

      m_moleculeList.push_back (data);
      imageList.append ("./vmd-1.8.7/ExscitechResources/DefaultTexture.tga");
      textList.append (QString (label.c_str ()));

    }
  }

  void
  ViewLigandGame::initWindow ()
  {

  }

  void
  ViewLigandGame::onUpdate ()
  {
  }

  bool
  ViewLigandGame::handleMouseInput (int screenX, int screenY, int button)
  {
    m_mouseLocation.set (screenX, screenY);
    fprintf (stderr, "Mouse Input\n");
    return true;
  }

  bool
  ViewLigandGame::handleMouseMove (int screenX, int screenY)
  {
    if (m_molecule != NULL)
    {
      int deltaX = screenX - m_mouseLocation.x;
      int deltaY = screenY - m_mouseLocation.y;
      m_mouseLocation.set (screenX, screenY);
      m_molecule->rotateWorld (0.25f * deltaX, Constants::WORLD_Y);
      m_molecule->rotateWorld (0.25f * deltaY, Constants::WORLD_X);
    }
    return true;
  }

  bool
  ViewLigandGame::handleMouseWheel (int delta)
  {
    m_camera->moveForward (0.033f * delta);
    return true;
  }

  void
  ViewLigandGame::drawGameGraphics ()
  {
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (m_molecule != NULL)
    {
      m_molecule->draw (m_camera);
    }
  }

  void
  ViewLigandGame::selectionFinished (void* selection)
  {

  }

  void*
  ViewLigandGame::onSelectionFinished (int selection)
  {
    return m_moleculeList[selection];
  }

  void
  ViewLigandGame::onSelectionChanged (int selection)
  {
    if (selection < m_moleculeList.size ())
    {
      if (m_molecule != NULL)
      {
        delete m_molecule;
        m_molecule = NULL;
      }
      m_molecule = MoleculeLoader::loadLabeledMolecule (
          m_moleculeList[selection]->getPdbFilePath ());
    }
  }
}
