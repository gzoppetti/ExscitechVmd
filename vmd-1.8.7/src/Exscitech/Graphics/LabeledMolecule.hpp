#ifndef LABELEDMOLECULE_HPP_
#define LABELEDMOLECULE_HPP_

#include <GL/glew.h>

#include "Exscitech/Graphics/LabeledAtoms.hpp"
#include "Exscitech/Graphics/Bonds.hpp"

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Display/Camera.hpp"

namespace Exscitech
{
  class LabeledMolecule : public Drawable
  {
  public:

    LabeledMolecule (LabeledAtoms* atoms, Bonds* bonds) :
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
    LabeledAtoms* m_atoms;
    Bonds* m_bonds;
  };
}
#endif
