#ifndef BALLANDSTICKMOLECULE_HPP_
#define BALLANDSTICKMOLECULE_HPP_

#include <GL/glew.h>

#include "Atoms.hpp"
#include "Bonds.hpp"

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Display/Camera.hpp"

namespace Exscitech
{
  class BallAndStickMolecule : public Drawable
  {
  public:

    BallAndStickMolecule (Atoms* atoms, Bonds* bonds) :
        m_atoms (atoms), m_bonds (bonds)
    {
    }

    virtual void
    draw (Camera* camera)
    {
      m_atoms->setTransform(getTransform());
      m_bonds->setTransform(getTransform());
      m_atoms->draw (camera);
      m_bonds->draw (camera);
    }

  private:
    Atoms* m_atoms;
    Bonds* m_bonds;
  };
}
#endif
