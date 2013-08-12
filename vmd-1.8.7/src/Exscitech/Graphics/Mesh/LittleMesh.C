#include <cstdio>
#include <GL/glew.h>

#include "DispCmds.h"

#include "Exscitech/Graphics/Mesh/LittleMesh.hpp"

namespace Exscitech
{

  using std::string;
  using std::vector;

  LittleMesh::LittleMesh (Displayable *parent) :
    Displayable (parent)
  {
  }

  LittleMesh::~LittleMesh ()
  {
  }

  void
  LittleMesh::drawMesh ()
  {
    glColor4f (1, 0, 0, 1.0f);
    glBegin (GL_TRIANGLES);
    glVertex3f (1, -1.5, 0);
    glVertex3f (-1, -1.5, 0);
    glVertex3f (0, 1.5, 0);
    glEnd ();
  }

  void
  LittleMesh::prepare ()
  {
    create_cmdlist ();
  }

  void
  LittleMesh::create_cmdlist ()
  {
   // DispCmdExscitechMesh dispCmdExscitechMesh;
   // dispCmdExscitechMesh.putdata (this, cmdList);
  }

}
