#ifndef NEWMESH_OBJ_BUILDER_HPP_
#define NEWMESH_OBJ_BUILDER_HPP_

#include <string>
#include <sstream>
#include <fstream>
#include <istream>
#include <vector>

// Included for ShaderNames
#include "Exscitech/Graphics/Drawable.hpp"

#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
  class Mesh;
  class MeshPart;
  class GeneralShaderProgram;
  class ObjMeshLoader
  {
  public:

    ObjMeshLoader ();

    Mesh
    *
    createMeshFromObjFile (const std::string& fileName, bool checkForDuplicates);

  private:

    void
    readMaterialFile (Mesh* mesh, const std::string& materialFile,
        const std::string& meshFile);

    void
    readMaterialFileHelper (std::istream& materialFile,
        const std::string& fileDirectory);

    Vector3f
    extractValues (std::istream& input);

    void
    initializeMesh (Mesh* newMesh, const std::vector<Vector3f>& positions,
        std::vector<Vector3f>& normals, const std::vector<Vector3f>& texCoords,
        std::stringstream& faceInfoy);

    MeshPart*
    initializeNewMeshPart (Mesh* mesh, const std::string& materialName,
        const std::string& faceInfo);

    void
    addVertexToMeshPart (MeshPart* currentMeshPart,
        const std::vector<Vector3f>& positions,
        const std::vector<Vector3f>& normals,
        const std::vector<Vector3f>& texCoords, const std::string& vertexSpecs);

    void
    createNormals (std::vector<Vector3f>& normals,
        const std::vector<Vector3f>& positions, const std::string& faceInfo);

    Vector3f
    generateFaceNormal (const std::vector<uint>& positionIndices,
        const std::vector<Vector3f>& positions);

    bool
    vertexSpecContainsTexCoord (const std::string& vertexSpec);

    bool
    vertexSpecContainsNormal (const std::string& vertexSpec);

    std::vector<uint>
    scanVertexSpec (const std::string& vertexSpec);

  private:

    size_t m_positionAttribId;
    size_t m_normalAttribId;
    size_t m_textCoordAttribId;

  };

}

#endif /* MESHOBJBUILDER_HPP_ */
