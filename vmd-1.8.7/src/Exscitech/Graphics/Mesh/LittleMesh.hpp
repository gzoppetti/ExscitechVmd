#ifndef LITTLE_MESH_HPP_
#define LITTLE_MESH_HPP_

#include <string>
#include <vector>

#include "Displayable.h"

namespace Exscitech
{

  class  LittleMesh : public Displayable
  {

  public:

    LittleMesh (Displayable *parent);

    ~LittleMesh ();


    void
    drawMesh ();

    /////Displayable Methods/////
    virtual void
    prepare ();

  private:

     void
     create_cmdlist ();

  };

}

#endif
