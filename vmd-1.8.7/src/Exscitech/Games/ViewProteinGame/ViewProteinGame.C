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

#include "Exscitech/Games/ViewProteinGame/ViewProteinGame.hpp"

#include "Exscitech/Graphics/SpaceFillMolecule.hpp"
#include "Exscitech/Graphics/BallAndStickMolecule.hpp"

#include "Exscitech/Graphics/MoleculeLoader.hpp"
#include "Exscitech/Graphics/FullQuad.hpp"

#include "Exscitech/Graphics/SSAO.hpp"

#include "Exscitech/Constants.hpp"

namespace Exscitech
{

  ViewProteinGame::ViewProteinGame (SelectionGameDelegate* delegate) :
      SelectionGame (delegate), m_molecule (NULL)
  {
    m_camera = new Camera (Vector4i (0, 0, ms_glSize.x, ms_glSize.y), 60.f,
        0.01f, 1000.f);
    m_camera->moveBackward (100);

    m_ssao = new SSAO (ms_glSize.x, ms_glSize.y);

    m_menuWindow->setWindowTitle (QString ("View Protein"));
  }

  ViewProteinGame::~ViewProteinGame ()
  {
    delete m_ssao;
    delete m_camera;
    delete m_molecule;

    for (MoleculeServerData* data : m_moleculeList)
    {
      delete data;
    }
  }

  void
  ViewProteinGame::initList (QStringList& imageList, QStringList& textList)
  {
    // TODO: Receive list from server.
    m_moleculeList.reserve (100);

    for (int i = 0; i < 10; ++i)
    {
      std::stringstream ss;
      ss << "Protein #" << i;
      std::string label = ss.str ();

      // TODO: Populate data structure from server's XML response
      MoleculeServerData* data = new MoleculeServerData ();
      data->setDownloadUrl ("TODO")->setId ("Id")->setMoleculeName ("Protein!")->setNotes (
          "Notes!")->setPdbFilePath (
          "./vmd-1.8.7/ExscitechResources/protein.pdb");

      m_moleculeList.push_back (data);
      imageList.append ("./vmd-1.8.7/ExscitechResources/DefaultTexture.tga");
      textList.append (QString (label.c_str ()));

    }
  }

  void
  ViewProteinGame::initWindow ()
  {

  }

  void
  ViewProteinGame::onUpdate ()
  {
  }

  bool
  ViewProteinGame::handleMouseInput (int screenX, int screenY, int button)
  {
    m_mouseLocation.set (screenX, screenY);
    fprintf (stderr, "Mouse Input\n");
    return true;
  }

  bool
  ViewProteinGame::handleMouseMove (int screenX, int screenY)
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
  ViewProteinGame::handleMouseWheel (int delta)
  {
    m_camera->moveForward (0.033f * delta);
    return true;
  }

  void
  ViewProteinGame::drawGameGraphics ()
  {
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (m_molecule != NULL)
    {
      m_ssao->enable ();
      m_molecule->draw (m_camera);
      m_ssao->disable ();
      m_ssao->draw (m_camera);
    }
  }

  void*
  ViewProteinGame::onSelectionFinished (int selection)
  {
    return m_moleculeList[selection];
  }

  void
  ViewProteinGame::onSelectionChanged (int selection)
  {
    if (selection >= 0 && selection < m_moleculeList.size ())
    {
      if (m_molecule != NULL)
      {
        delete m_molecule;
        m_molecule = NULL;
      }
      m_molecule = MoleculeLoader::loadSpaceFillMolecule (
          m_moleculeList[selection]->getPdbFilePath ());
    }
  }
}
