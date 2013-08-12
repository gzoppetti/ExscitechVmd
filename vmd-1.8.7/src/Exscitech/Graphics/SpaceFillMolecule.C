#include <map>

#include "Exscitech/Graphics/SpaceFillMolecule.hpp"
#include "Exscitech/Constants.hpp"

namespace Exscitech
{
  SpaceFillMolecule::SpaceFillMolecule (const std::vector<Vector3f>& positions,
      const std::vector<AtomicName>& atomNames) :
      m_program ("./vmd-1.8.7/ExscitechResources/Shaders/SphereShader.vsh",
          "./vmd-1.8.7/ExscitechResources/Shaders/SphereShader.fsh",
          "./vmd-1.8.7/ExscitechResources/Shaders/SphereShader.gsh")
  {
    unsigned int numNames = atomNames.size ();

    std::map<AtomicName, std::vector<Vector3f> > indexMap;

    for (unsigned int i = 0; i < numNames; ++i)
    {
      indexMap[atomNames[i]].push_back (positions[i]);
    }

    for (std::pair<const AtomicName, std::vector<Vector3f> >& pair : indexMap)
    {
      AtomGroup& group = m_atomMap[pair.first];
      group.numPoints = pair.second.size ();
      determineDetail (pair.first, group.detail);
      glGenBuffers (1, &group.bufferHandle);
      glBindBuffer (GL_ARRAY_BUFFER, group.bufferHandle);
      glBufferData (GL_ARRAY_BUFFER, sizeof(Vector3f) * pair.second.size (),
          &pair.second[0], GL_STATIC_DRAW);
      glBindBuffer (GL_ARRAY_BUFFER, 0);
    }
  }

  void
  SpaceFillMolecule::determineDetail (const AtomicName& name, Vector4f& detail)
  {
    detail.set(MoleculeLoader::getAtomicDetailFromName(name), Constants::DEFAULT_RADIUS * Constants::SPACE_FILL_RADIUS_SCALE);
  }

  void
  SpaceFillMolecule::draw (Camera* camera)
  {
    m_program.enable ();
    m_program.setUniform (m_program.getUniformLocation ("g_world"),
        getTransform4x4 ());
    m_program.setUniform (m_program.getUniformLocation ("g_view"),
        camera->getView ());
    m_program.setUniform (m_program.getUniformLocation ("g_projection"),
        camera->getProjection ());

    for (std::pair<const AtomicName, AtomGroup>& pair : m_atomMap)
    {
      m_program.setAttribPointer (pair.second.bufferHandle,
          m_program.getAttribLocation ("g_point"), 3, GL_FLOAT, GL_FALSE,
          sizeof(Vector3f), 0, 0);
      m_program.setUniform (m_program.getUniformLocation ("g_detail"),
          pair.second.detail);
      glDrawArrays (GL_POINTS, 0, pair.second.numPoints);
    }

    m_program.disableAttribute ("g_point");
    m_program.disable ();
  }
}
