#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <istream>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "Exscitech/Constants.hpp"

#include "Exscitech/Graphics/Lighting/Material.hpp"
#include "Exscitech/Graphics/Lighting/MaterialLibrary.hpp"
#include "Exscitech/Graphics/Lighting/Texture.hpp"

#include "Exscitech/Graphics/Mesh/ObjMeshLoader.hpp"
#include "Exscitech/Graphics/Mesh/Mesh.hpp"
#include "Exscitech/Graphics/Mesh/MeshPart.hpp"
#include "Exscitech/Graphics/Mesh/VertexAttribute.hpp"
#include "Exscitech/Graphics/Mesh/VertexDescriptor.hpp"

#include "Exscitech/Math/Vector3.hpp"

using std::string;
using std::vector;
using std::ifstream;
using std::stringstream;
using std::istream;
using std::getline;

namespace Exscitech
{
  ObjMeshLoader::ObjMeshLoader ()
  {
  }

  Mesh*
  ObjMeshLoader::createMeshFromObjFile (const string& meshFileName,
      bool checkForDuplicates)
  {
    Mesh* newMesh = new Mesh ("Mesh from " + meshFileName);

    ifstream meshFile;
    meshFile.open (meshFileName.c_str ());
    if (!meshFile)
    {
      fprintf (stderr, "Error, could not read OBJ file: %s.\n",
          meshFileName.c_str ());
      return (NULL);
    }

    //    MaterialLibrary materialLibrary;
    vector<Vector3f> positions;
    vector<Vector3f> normals;
    vector<Vector3f> textureCoords;

    stringstream faceInfo;

    string nextLine;
    string firstToken;
    while (getline (meshFile, nextLine))
    {
      stringstream lineReader (nextLine);
      lineReader >> firstToken;
      if (!lineReader)
      {
        continue;
      }
      if (firstToken.compare ("mtllib") == 0)
      {
        string materialFile;
        lineReader >> materialFile;
        readMaterialFile (newMesh, materialFile, meshFileName);
      }
      else if (firstToken.compare ("v") == 0)
      {
        Vector3f position = extractValues (lineReader);
        positions.push_back (position);
      }
      else if (firstToken.compare ("vt") == 0)
      {
        Vector3f textureCoord = extractValues (lineReader);
        textureCoords.push_back (textureCoord);
      }
      else if (firstToken.compare ("vn") == 0)
      {
        Vector3f normal = extractValues (lineReader);
        normals.push_back (normal);
      }
      else if (firstToken.compare ("f") == 0
          || firstToken.compare ("usemtl") == 0)
      {
        faceInfo << nextLine << " ";
      }
      else
      {
        //fprintf (stderr, "Skipping line: |%s|\n", nextLine.c_str ());
      }
    }

    meshFile.close ();
    initializeMesh (newMesh, positions, normals, textureCoords, faceInfo);
    //newMesh->printBuffers ();

    return (newMesh);
  }

  void
  ObjMeshLoader::readMaterialFile (Mesh* mesh, const std::string& materialFile,
      const std::string& meshFile)
  {
    namespace fs = boost::filesystem;

    fs::path path (meshFile);
    fs::path directory = path.parent_path ();
    std::string materialFileDirectory = directory.string () + "/";
    std::string materialPath = materialFileDirectory + materialFile;
    fprintf (stderr, "%s\n", materialPath.c_str ());

    std::string nextFileName;
    std::ifstream materialFileIn;

    materialFileIn.open (materialPath.c_str ());
    if (!materialFileIn)
    {
      fprintf (stderr, "Error: could not open material file.\n");
    }
    else
    {
      readMaterialFileHelper (materialFileIn, materialFileDirectory);
    }
    materialFileIn.close ();
  }

  void
  ObjMeshLoader::readMaterialFileHelper (std::istream& materialFile,
      const std::string& fileDirectory)
  {
    Exscitech::Material* material = NULL;
    std::string line;
    while (std::getline (materialFile, line))
    {
      std::stringstream lineStream (line);
      std::string token;
      lineStream >> token;

      if (token == "newmtl")
      {
        std::string materialName;
        lineStream >> materialName;
        material = Material::create (materialName);
        if (material == NULL)
        {
          return;
        }
      }
      else if (token == "Ka")
      {
        Vector3f color;
        lineStream >> color.x >> color.y >> color.z;
        material->setAmbientColor (color);
      }
      else if (token == "Kd")
      {
        Vector3f color;
        lineStream >> color.x >> color.y >> color.z;
        material->setDiffuseColor (color);
      }
      else if (token == "Ks")
      {
        Vector3f color;
        lineStream >> color.x >> color.y >> color.z;
        material->setSpecularColor (color);
      }
      else if (token == "illum")
      {
        // Illumination model -- not handling
      }
      else if (token == "Ns")
      {
        float shininess;
        lineStream >> shininess;
        material->setShininess (shininess);
      }
      else if (token == "map_Kd")
      {
        std::string textureFile;
        lineStream >> textureFile;
        std::string textureFileWithDirectory = fileDirectory + textureFile;

        fprintf (stderr, "%s\n", textureFileWithDirectory.c_str ());
        Texture* texture = Texture::create (textureFile,
            textureFileWithDirectory);
        material->setTexture (texture);
      }
    }
  }
  Vector3f
  ObjMeshLoader::extractValues (istream& input)
  {
    Vector3f valuesVec;
    float value;
    for (int i = 0; i < 3; ++i)
    {
      input >> value;
      valuesVec.coords[i] = value;
    }
    return (valuesVec);
  }

  void
  ObjMeshLoader::initializeMesh (Mesh* newMesh,
      const vector<Vector3f>& positions, vector<Vector3f>& normals,
      const vector<Vector3f>& texCoords, stringstream& faceInfo)
  {

    if (normals.empty ())
    {
      // Pass a copy of faceInfo string
      createNormals (normals, positions, faceInfo.str ());
    }

    MeshPart* currentMeshPart;

    bool newMeshPartNeeded = true;

    string newMeshPartMaterial = "ObjLoaderDefault";
    Material::create (newMeshPartMaterial);

    string nextToken;
    while (faceInfo >> nextToken)
    {
      if (nextToken.compare ("usemtl") == 0)
      {
        faceInfo >> newMeshPartMaterial;
        newMeshPartNeeded = true;
      }
      else if (nextToken.compare ("f") == 0)
      {
        // Get the three vertex indexes specified for the triangular face
        vector<string> vertexSpecs (3);
        for (int i = 0; i < 3; ++i)
        {
          faceInfo >> vertexSpecs[i];
        }

        if (newMeshPartNeeded)
        {
          currentMeshPart = initializeNewMeshPart (newMesh, newMeshPartMaterial,
              vertexSpecs[0]);
          newMesh->addMeshPart (currentMeshPart);
          newMeshPartNeeded = false;
        }

        for (int i = 0; i < 3; ++i)
        {
          addVertexToMeshPart (currentMeshPart, positions, normals, texCoords,
              vertexSpecs[i]);
        }
      }
    }

    newMesh->initializeMeshParts ();

  }

  MeshPart*
  ObjMeshLoader::initializeNewMeshPart (Mesh* mesh, const string& materialName,
      const string& faceInfo)
  {
    // Create vertex descriptor for this mesh part
    string meshPartName = "Mesh Part with " + materialName + " material";
    VertexDescriptor meshDescriptor (meshPartName + " descriptor");

    VertexAttribute::Type type = VertexAttribute::FLOAT3;
    VertexAttribute::Usage usage = VertexAttribute::POSITION;
    m_positionAttribId = meshDescriptor.addAttribute (usage, type);

    type = VertexAttribute::FLOAT3;
    usage = VertexAttribute::NORMAL;
    m_normalAttribId = meshDescriptor.addAttribute (usage, type);

    if (vertexSpecContainsTexCoord (faceInfo))
    {
      type = VertexAttribute::FLOAT2;
      usage = VertexAttribute::TEX_COORD0;
      m_textCoordAttribId = meshDescriptor.addAttribute (usage, type);
    }

    fprintf (stderr, "Looking up %s \n", materialName.c_str ());
    Material* material = Material::lookup (materialName);

    if (material == NULL)
    {
      fprintf (stderr, "Material not found at meshpart creation!\n");
    }

    MeshPart* meshPart = new MeshPart (meshPartName, meshDescriptor, material);

    return (meshPart);
  }

  bool
  ObjMeshLoader::vertexSpecContainsTexCoord (const string& vertexSpec)
  {
    vector<string> indexTokens;
    //   posIndex
    //   posIndex/texIndex
    //   posIndex//normIndex
    //   posIndex/texIndex/normIndex
    boost::split (indexTokens, vertexSpec, boost::is_any_of ("/"));
    bool containsTexCoord = false;
    if (indexTokens.size () == 2
        || (indexTokens.size () == 3 && !indexTokens[1].empty ()))
    {
      // There are 2 indices
      //   or (there are 3 and the 2nd isn't missing)
      containsTexCoord = true;
    }
    return (containsTexCoord);
  }

  bool
  ObjMeshLoader::vertexSpecContainsNormal (const string& vertexSpec)
  {
    vector<string> indexTokens;
    //   posIndex
    //   posIndex/texIndex
    //   posIndex//normIndex
    //   posIndex/texIndex/normIndex
    boost::split (indexTokens, vertexSpec, boost::is_any_of ("/"));
    bool containsNormal = false;
    if (indexTokens.size () == 3)
    {
      containsNormal = true;
    }
    return (containsNormal);
  }

  vector<uint>
  ObjMeshLoader::scanVertexSpec (const string& vertexSpec)
  {
    vector<string> indexTokens;
    boost::split (indexTokens, vertexSpec, boost::is_any_of ("/"));
    vector<uint> indices (3, UINT_MAX);
    for (size_t i = 0; i < indexTokens.size (); ++i)
    {
      if (!indexTokens[i].empty ())
      {
        indices[i] = boost::lexical_cast<uint> (indexTokens[i]);
      }
    }

    return (indices);
  }

  void
  ObjMeshLoader::addVertexToMeshPart (MeshPart* currentMeshPart,
      const vector<Vector3f>& positions, const vector<Vector3f>& normals,
      const vector<Vector3f>& texCoords, const string& vertexSpecs)
  {
    vector<uint> indices = scanVertexSpec (vertexSpecs);
    // Position/texCoord/normal is in indices
    uint positionIndex = indices[0] - 1;

    VertexDescriptor partDescriptor = currentMeshPart->getVertexDescriptor ();
    Vertex vertex (partDescriptor);
    vertex.setAttribute (m_positionAttribId, &positions[positionIndex][0]);

    if (indices[1] != UINT_MAX)
    {
      uint textureIndex = indices[1] - 1;
      vertex.setAttribute (m_textCoordAttribId, &texCoords[textureIndex][0]);
    }

    uint normalIndex = positionIndex;
    if (indices[2] != UINT_MAX)
    {
      normalIndex = indices[2] - 1;
    }
    vertex.setAttribute (m_normalAttribId, &normals[normalIndex][0]);
    uint vertexIndex = currentMeshPart->addVertex (vertex);

    currentMeshPart->addIndex (vertexIndex);
  }

  void
  ObjMeshLoader::createNormals (vector<Vector3f>& normals,
      const vector<Vector3f>& positions, const string& faceInfo)
  {
    normals.resize (positions.size (), Vector3f (0.0f));

    stringstream faceInfoReader (faceInfo);
    string nextToken;
    while (faceInfoReader >> nextToken)
    {
      if (nextToken.compare ("f") == 0)
      {
        // It's a face, so calculate & average in normal for it
        // Get indices (into positions vector) which make up face
        vector<uint> positionIndices (3);
        string indexString;
        for (int i = 0; i < 3; ++i)
        {
          faceInfoReader >> nextToken;
          indexString = nextToken.substr (0, nextToken.find ('/'));
          positionIndices[i] = boost::lexical_cast<uint> (indexString) - 1;
        }

        Vector3f faceNormal = generateFaceNormal (positionIndices, positions);
        for (int i = 0; i < 3; ++i)
        {
          // the normal associated with each position
          // will be average of the normals of all
          // faces of which that position is part
          // so here, average in normal calculated
          // for specific face
          normals[positionIndices[i]] += faceNormal;
        }
      }
    }

    // Normalize normals for faster rendering
    for (size_t i = 0; i < normals.size (); ++i)
    {
      normals[i].normalize ();
    }
  }

  Vector3f
  ObjMeshLoader::generateFaceNormal (const vector<uint>& positionIndices,
      const vector<Vector3f>& positions)
  {
    // Vertices, oriented CCW
    Vector3f v1 = positions[positionIndices[0]];
    Vector3f v2 = positions[positionIndices[1]];
    Vector3f v3 = positions[positionIndices[2]];

    // Take two sides (v2v1 and v2v3) and cross to get normal
    v1 -= v2;
    v3 -= v2;
    // OpenGL is right-handed
    Vector3f normal = v3.cross (v1);
    return (normal);
  }
}
