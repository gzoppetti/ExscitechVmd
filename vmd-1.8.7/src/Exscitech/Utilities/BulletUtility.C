#include <bullet/BulletSoftBody/btSoftBodyHelpers.h>
#include <bullet/btBulletDynamicsCommon.h>
#include <bullet/btBulletCollisionCommon.h>

#include "Exscitech/Graphics/VolmapExscitech.hpp"
#include "Exscitech/Graphics/Mesh/Mesh.hpp"
#include "Exscitech/Graphics/Mesh/MeshPart.hpp"
#include "Exscitech/Utilities/IndexedTriangleList.hpp"
#include "Exscitech/Utilities/BulletUtility.hpp"

namespace Exscitech
{
  BulletUtility::ShapeMap BulletUtility::ms_shapeMap;

  BulletUtility::BulletUtility ()
  {
  }

  void
  BulletUtility::clearShapeMap ()
  {
    for (ShapeMapConstIter iter = ms_shapeMap.begin ();
        iter != ms_shapeMap.end (); ++iter)
    {
      delete iter->second;
    }
    ms_shapeMap.clear ();

  }
  btCollisionShape*
  BulletUtility::acquireBtCollisionShape (const std::string& shapeName)
  {
    ShapeMapConstIter iter = ms_shapeMap.find (shapeName);
    btCollisionShape* shape = NULL;
    if (iter != ms_shapeMap.end ())
    {
      shape = iter->second;
    }
    return shape;
  }

  bool
  BulletUtility::addShapeToMap (const std::string& shapeName,
      btCollisionShape* shape, bool overrideIfFound)
  {
    ShapeMapConstIter iter = ms_shapeMap.find (shapeName);

    if (iter == ms_shapeMap.end ())
    {
      ms_shapeMap.insert (std::make_pair (shapeName, shape));
      return true;
    }
    else if (overrideIfFound)
    {
      ms_shapeMap.erase (shapeName);
      ms_shapeMap.insert (make_pair (shapeName, shape));
      return true;
    }
    else
    {
      fprintf (stderr, "SHAPE NOT ADDED TO MAP \n");
      return false;
    }

  }

  bool
  BulletUtility::createBulletConcaveTriMeshFromVolmap (
      const std::string& shapeName, VolmapExscitech* volmap,
      bool overrideIfFound)
  {
    ShapeMapConstIter iter = BulletUtility::ms_shapeMap.find (shapeName);

    if (iter == BulletUtility::ms_shapeMap.end () || overrideIfFound)
    {
      float* vertices;
      int* indices;
      int numVertices;
      int numIndices;
      volmap->getData (vertices, indices, numVertices, numIndices);
      //int numTriangles, int *triangleIndexBase, int triangleIndexStride, int numVertices, btScalar *vertexBase, int vertexStride
      btTriangleIndexVertexArray* array = new btTriangleIndexVertexArray (
          numIndices / 3, indices, 3 * sizeof(int), numVertices, vertices,
          3 * sizeof(float));

      btCollisionShape* shape = new btBvhTriangleMeshShape (array, true);
      if (iter == ms_shapeMap.end ())
      {
        BulletUtility::ms_shapeMap.insert (std::make_pair (shapeName, shape));
      }
      else if (overrideIfFound)
      {
        ms_shapeMap.erase (shapeName);
        ms_shapeMap.insert (make_pair (shapeName, shape));
      }
      return true;
    }
    else
      return false;
  }

  bool
  BulletUtility::createBulletConcaveTriMeshFromMeshInfo (
      const std::string& shapeName, Mesh* mesh, bool overrideIfFound)
  {
    ShapeMapConstIter iter = BulletUtility::ms_shapeMap.find (shapeName);

    if (iter == BulletUtility::ms_shapeMap.end () || overrideIfFound)
    {
      btTriangleIndexVertexArray* array = new btTriangleIndexVertexArray ();

      MeshPart* part;

      for (size_t i = 0; i < mesh->getNumberOfParts (); ++i)
      {
        part = mesh->getMeshPart (i);
        std::vector<float>& vertices = part->getVertexVector ();
        std::vector<uint>& indices = part->getIndexVector ();
        int indexStride = 3 * sizeof(uint);
        int numTriangles = indices.size () / 3;
        int vertexSizeInFloats = part->getVertexSizeInFloats ();
        int numVertices = vertices.size () / vertexSizeInFloats;
        int vertexStride = vertexSizeInFloats * sizeof(float);

        btIndexedMesh mesh = createBtIndexedMesh (vertices, indices,
            indexStride, numTriangles, vertexSizeInFloats, numVertices,
            vertexStride);
        array->addIndexedMesh (mesh);
      }

      btCollisionShape* shape = new btBvhTriangleMeshShape (array, true);

      if (iter == ms_shapeMap.end ())
      {
        BulletUtility::ms_shapeMap.insert (make_pair (shapeName, shape));
      }
      else if (overrideIfFound)
      {
        ms_shapeMap.erase (shapeName);
        ms_shapeMap.insert (make_pair (shapeName, shape));
      }
      return true;
    }
    else
      return false;
  }

  bool
  BulletUtility::createBulletConvexTriMeshFromMeshInfo (
      const std::string& shapeName, Mesh* mesh, bool overrideIfFound)
  {
    ShapeMapConstIter iter = BulletUtility::ms_shapeMap.find (shapeName);

    if (iter == BulletUtility::ms_shapeMap.end () || overrideIfFound)
    {
      btTriangleIndexVertexArray* array = new btTriangleIndexVertexArray ();

      MeshPart* part;

      for (size_t i = 0; i < mesh->getNumberOfParts (); ++i)
      {
        part = mesh->getMeshPart (i);
        std::vector<float>& vertices = part->getVertexVector ();
        std::vector<uint>& indices = part->getIndexVector ();
        int indexStride = 3 * sizeof(uint);
        int numTriangles = indices.size () / 3;
        int vertexSizeInFloats = part->getVertexSizeInFloats ();
        int numVertices = vertices.size () / vertexSizeInFloats;
        int vertexStride = vertexSizeInFloats * sizeof(float);

        btIndexedMesh mesh = createBtIndexedMesh (vertices, indices,
            indexStride, numTriangles, vertexSizeInFloats, numVertices,
            vertexStride);
        array->addIndexedMesh (mesh);
      }

      btCollisionShape* shape = new btConvexTriangleMeshShape (array, true);
      if (iter == ms_shapeMap.end ())
      {
        BulletUtility::ms_shapeMap.insert (make_pair (shapeName, shape));
      }
      else if (overrideIfFound)
      {
        ms_shapeMap.erase (shapeName);
        ms_shapeMap.insert (make_pair (shapeName, shape));
      }
      return true;
    }
    else
      return false;
  }

  bool
  BulletUtility::createBulletConvexHullFromMeshInfo (
      const std::string& shapeName, Mesh* mesh, bool overrideIfFound)
  {
    ShapeMapConstIter iter = BulletUtility::ms_shapeMap.find (shapeName);

    if (iter == BulletUtility::ms_shapeMap.end () || overrideIfFound)
    {
      MeshPart* part;

      btConvexHullShape * hullShape = new btConvexHullShape ();
      for (size_t i = 0; i < mesh->getNumberOfParts (); ++i)
      {
        part = mesh->getMeshPart (i);
        std::vector<float>& vertices = part->getVertexVector ();
        int vertexSizeInFloats = part->getVertexSizeInFloats ();

        for (size_t j = 0; j < vertices.size (); j += vertexSizeInFloats)
        {
          hullShape->addPoint (
              btVector3 (vertices[j], vertices[j + 1], vertices[j + 2]));
        }
      }

      if (iter == ms_shapeMap.end ())
      {
        BulletUtility::ms_shapeMap.insert (make_pair (shapeName, hullShape));
      }
      else if (overrideIfFound)
      {
        ms_shapeMap.erase (shapeName);
        ms_shapeMap.insert (make_pair (shapeName, hullShape));
      }
      return true;
    }
    else
      return false;
  }

  btIndexedMesh
  BulletUtility::createBtIndexedMesh (std::vector<float>& vertices,
      std::vector<uint>& indices, int indexStride, int numTriangles,
      int vertexSizeInFloats, int numVertices, int vertexStride)
  {
    btIndexedMesh mesh;
    mesh.m_indexType = PHY_INTEGER;
    mesh.m_numTriangles = numTriangles;
    mesh.m_numVertices = numVertices;
    mesh.m_triangleIndexBase = reinterpret_cast<unsigned char*> (&indices[0]);
    mesh.m_triangleIndexStride = indexStride;
    mesh.m_vertexBase = reinterpret_cast<unsigned char*> (&vertices[0]);
    mesh.m_vertexStride = vertexStride;
    return mesh;
  }

  bool
  BulletUtility::createBulletConcaveShapeFromVertices (
      const std::string& shapeName, const std::vector<Vector3f>& vertices,
      bool overrideIfFound)
  {
    ShapeMapConstIter iter = BulletUtility::ms_shapeMap.find (shapeName);

    if (iter == BulletUtility::ms_shapeMap.end () || overrideIfFound)
    {
      btTriangleMesh* mesh = new btTriangleMesh (true, false);

      for (size_t i = 0; i < vertices.size ();)
      {
        mesh->addTriangle (vertices[i].toBtVector3 (),
            vertices[i + 1].toBtVector3 (), vertices[i + 2].toBtVector3 (),
            false);
        i += 3;
      }

      btCollisionShape * shape = new btBvhTriangleMeshShape (mesh, true, true);

      if (iter == ms_shapeMap.end ())
      {
        BulletUtility::ms_shapeMap.insert (std::make_pair (shapeName, shape));
      }
      else if (overrideIfFound)
      {
        ms_shapeMap.erase (shapeName);
        ms_shapeMap.insert (make_pair (shapeName, shape));
      }
      return true;
    }
    else
      return false;
  }

  bool
  BulletUtility::createBulletMultiSphereShape (const std::string& shapeName,
      std::vector<Vector3f>& positions, std::vector<float>& radii,
      bool overrideIfFound)
  {
    ShapeMapConstIter iter = ms_shapeMap.find (shapeName);

    if (iter == ms_shapeMap.end ())
    {

      std::vector<btVector3> btPositions;
      for (size_t i = 0; i < positions.size (); ++i)
      {
        btPositions.push_back (positions[i].toBtVector3 ());
      }
      fprintf (stderr, "Creating Molecule Shape: %ld %ld\n",
          btPositions.size (), radii.size ());
      btCollisionShape * shape = new btMultiSphereShape (&btPositions[0],
          &radii[0], btPositions.size ());
      fprintf (stderr, "End Molecule Shape\n");
      ms_shapeMap.insert (std::make_pair (shapeName, shape));
      return true;

    }
    else if (overrideIfFound)
    {

      std::vector<btVector3> btPositions (positions.size ());
      for (size_t i = 0; i < positions.size (); ++i)
      {
        btPositions.push_back (positions[i].toBtVector3 ());
      }

      btCollisionShape * shape = new btMultiSphereShape (&btPositions[0],
          &radii[0], positions.size ());

      ms_shapeMap.erase (shapeName);
      ms_shapeMap.insert (make_pair (shapeName, shape));
      return true;
    }
    else
      return false;
  }

  btRigidBody*
  BulletUtility::createRigidBody (const std::string& shapeName, float mass,
      const Vector3f& startingPosition, const Vector3f& initialVelocity)
  {
    btRigidBody* body = NULL;

    ShapeMapConstIter iter = ms_shapeMap.find (shapeName);

    if (iter != ms_shapeMap.end ())
    {
      btVector3 initialInertia (0, 0, 0);
      iter->second->calculateLocalInertia (mass, initialInertia);
      btDefaultMotionState* defaultMotionState = new btDefaultMotionState (
          btTransform (
              btQuaternion (0, 0, 0, 1),
              btVector3 (startingPosition.x, startingPosition.y,
                  startingPosition.z)));

      body = new btRigidBody (mass, defaultMotionState, iter->second,
          initialInertia);
      btVector3 btVector = initialVelocity.toBtVector3 ();
      body->setLinearVelocity (btVector);
    }
    else
    {
      fprintf (stderr, "Returning NULL from createRigidBody!");
    }
    return body;
  }

  btRigidBody*
  BulletUtility::createRigidBody (btCollisionShape* shape, float mass,
      const Vector3f& startingPosition, const Vector3f& initialVelocity)
  {
    btVector3 initialInertia (0, 0, 0);
    shape->calculateLocalInertia (mass, initialInertia);
    btDefaultMotionState* defaultMotionState = new btDefaultMotionState (
        btTransform (
            btQuaternion (0, 0, 0, 1),
            btVector3 (startingPosition.x, startingPosition.y,
                startingPosition.z)));

    btRigidBody* body = new btRigidBody (mass, defaultMotionState, shape,
        initialInertia);

    btVector3 btVector = initialVelocity.toBtVector3 ();
    body->setLinearVelocity (btVector);

    return body;
  }

  void
  BulletUtility::createStaticSpheresFromPositions (btSphereShape* sphere,
      const std::vector<Vector3f>& positions, std::vector<btRigidBody*>& bodies)
  {
    btRigidBody* body;
    for (size_t i = 0; i < positions.size (); ++i)
    {
      body = createRigidBody (sphere, 0, positions[i], Vector3f (0, 0, 0));
      bodies.push_back (body);
    }
  }

}
