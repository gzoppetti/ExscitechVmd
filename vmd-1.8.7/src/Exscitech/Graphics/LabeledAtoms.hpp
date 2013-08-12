#ifndef LABELED_ATOMS_HPP_
#define LABELED_ATOMS_HPP_

#include <GL/glut.h>
#include <QtCore/QString>
#include <QtGui/QFont>

#include <cstdio>
#include "Drawable.hpp"

#include "Exscitech/Games/GameController.hpp"
#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Constants.hpp"

#include "Exscitech/Graphics/Lighting/Texture.hpp"
#include "Exscitech/Graphics/FullQuad.hpp"

#include "Exscitech/Utilities/TextUtility.hpp"

#include "Exscitech/Graphics/AtomicName.hpp"
#include "Exscitech/Graphics/MoleculeLoader.hpp"

namespace Exscitech
{
  class LabeledAtoms : public Drawable
  {
  public:

    LabeledAtoms (const std::vector<AtomicName>& names,
        const std::vector<Vector3f>& positions, float atomScale = 1) :
        m_program (
            "./vmd-1.8.7/ExscitechResources/Shaders/TexturedSphereShader.vsh",
            "./vmd-1.8.7/ExscitechResources/Shaders/TexturedSphereShader.fsh",
            "./vmd-1.8.7/ExscitechResources/Shaders/TexturedSphereShader.gsh")
    {
      std::map<AtomicName, std::vector<Vector3f> > vertexMap;

      unsigned int numNames = names.size ();
      for (unsigned int i = 0; i < numNames; ++i)
      {
        vertexMap[names[i]].push_back (positions[i]);
      }

      for (std::pair<const AtomicName, std::vector<Vector3f> >& pair : vertexMap)
      {
        AtomGroup& group = m_atomMap[pair.first];
        group.numPoints = pair.second.size ();
        determineDetail (atomScale, pair.first, group.detail);
        group.texId = createTextureForAtom (pair.first);
        glGenBuffers (1, &group.bufferHandle);
        glBindBuffer (GL_ARRAY_BUFFER, group.bufferHandle);
        glBufferData (GL_ARRAY_BUFFER, sizeof(Vector3f) * pair.second.size (),
            &pair.second[0], GL_STATIC_DRAW);
        glBindBuffer (GL_ARRAY_BUFFER, 0);
      }
    }

    void
    determineDetail (int atomScale, const AtomicName& name, Vector4f& detail)
    {
      detail.set(MoleculeLoader::getAtomicDetailFromName(name), atomScale * Constants::DEFAULT_RADIUS);
    }

    unsigned int
    createTextureForAtom (const AtomicName& name)
    {
      // STUFF
      Texture* sphereTex = Texture::create ("Sphere",
          "./vmd-1.8.7/ExscitechResources/Sphere.png");
      FullQuad quad;
      unsigned int completeTexId;

      quad.setTexture (sphereTex->getTextureID ());

      unsigned int fboId;
      glGenFramebuffers (1, &fboId);
      glActiveTexture (GL_TEXTURE0);
      glGenTextures (1, &completeTexId);

      glBindTexture (GL_TEXTURE_2D, completeTexId);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, 32, 32, 0, GL_RGBA,
          GL_UNSIGNED_BYTE, NULL);

      int oldBinding;
      glGetIntegerv (GL_FRAMEBUFFER_BINDING, &oldBinding);
      glBindFramebuffer (GL_FRAMEBUFFER, fboId);
      glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
          GL_TEXTURE_2D, completeTexId, 0);

      glClearColor (0, 0, 0, 0);
      glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, 32, 32);
      quad.draw(2);
      glColor3f(0,0,0);

      int x = 10;
      if (name.name[1] != '\0')
        x = 5;
      PrintText(x, 8, name.name);
      glPopAttrib();
      glDeleteFramebuffers(1, &fboId);

      glBindFramebuffer(GL_FRAMEBUFFER, oldBinding);

      return completeTexId;
    }

    virtual void
    draw (Camera* camera)
    {
      m_program.enable ();
      m_program.setUniform (m_program.getUniformLocation ("g_world"),
          getTransform4x4 ());
      m_program.setUniform (m_program.getUniformLocation ("g_view"),
          camera->getView ());
      m_program.setUniform (m_program.getUniformLocation ("g_projection"),
          camera->getProjection ());

      glActiveTexture(GL_TEXTURE0);
      m_program.setUniform(m_program.getUniformLocation("g_sampler"), 0);
      for (std::pair<const AtomicName, AtomGroup>& pair : m_atomMap)
      {
        m_program.setAttribPointer (pair.second.bufferHandle,
            m_program.getAttribLocation ("g_point"), 3, GL_FLOAT, GL_FALSE,
            sizeof(Vector3f), 0, 0);

        m_program.setUniform (m_program.getUniformLocation ("g_detail"),
            pair.second.detail);
        glBindTexture(GL_TEXTURE_2D, pair.second.texId);

        glDrawArrays (GL_POINTS, 0, pair.second.numPoints);
      }

      m_program.disableAttribute ("g_point");
      m_program.disable ();
    }

  private:
    struct AtomGroup
    {
      GLuint bufferHandle;
      GLuint texId;
      Vector4f detail;
      int numPoints;
    };

    std::map<AtomicName, AtomGroup> m_atomMap;

  private:
    ShaderProgram m_program;
  };
}
#endif
