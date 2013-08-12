#ifndef VOLMAPEXSCITECH_HPP_
#define VOLMAPEXSCITECH_HPP_

#include <GL/glew.h>

#include "VolumetricData.h"
#include "VMDApp.h"
#include "Molecule.h"
#include "Isosurface.h"
#include "Scene.h"
#include "MoleculeList.h"
#include "UIText.h"

#include "Exscitech/Graphics/VolmapPiece.hpp"
#include "Exscitech/Games/GameController.hpp"

#include "Exscitech/Graphics/Shaders/ShaderProgram.hpp"
#include "Exscitech/Graphics/Lighting/Material.hpp"

#include "Exscitech/Graphics/Drawable.hpp"

namespace Exscitech
{
  class VolmapExscitech : public Drawable
  {
  public:

    VolmapExscitech (const std::string& moleculeFile);

    virtual void
    draw (Camera* camera);

    void
    printArrays ();

    static
    void
    cache (float* cvn, int* indices, int numVerts, int numIndices);

    void
    getData (float*& vertices, int*& indices, int& numVertices,
        int& numIndices);

    std::vector<VolmapPiece*>
    split (unsigned int numPieces);

  private:

    std::vector<Vector3f> m_vertices;
    std::vector<Vector3f> m_normals;
    std::vector<int> m_indices;
    ShaderProgram m_program;
    Material m_material;
    LightUniformManager m_lightUniformManager;

    static std::vector<Vector3f>* ms_cachedVertices;
    static std::vector<Vector3f>* ms_cachedNormals;
    static std::vector<int>* ms_cachedIndices;

  };
}

#endif /* VOLMAPEXSCITECH_HPP_ */
