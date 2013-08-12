#ifndef SPACEFILLMOLECULE_HPP_
#define SPACEFILLMOLECULE_HPP_

#include "Exscitech/Graphics/Drawable.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Graphics/MoleculeLoader.hpp"

namespace Exscitech
{
  class SpaceFillMolecule : public Drawable
  {
  public:

    SpaceFillMolecule (const std::vector<Vector3f>& positions,
        const std::vector<AtomicName>& atomNames);

    void
    determineDetail (const AtomicName& name, Vector4f& detail);

    void
    draw(Camera* camera);

  private:

    struct AtomGroup
    {
      GLuint bufferHandle;
      Vector4f detail;
      unsigned int numPoints;
    };
    ShaderProgram m_program;
    std::map<AtomicName, AtomGroup> m_atomMap;
  };
}
#endif
