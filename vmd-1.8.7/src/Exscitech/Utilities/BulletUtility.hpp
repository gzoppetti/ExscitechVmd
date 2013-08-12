#ifndef BULLETUTILITY_HPP_
#define BULLETUTILITY_HPP_

#include <map>
#include <string>

#include <bullet/BulletSoftBody/btSoftBodyHelpers.h>
#include <bullet/btBulletDynamicsCommon.h>
#include <bullet/btBulletCollisionCommon.h>

#include "Exscitech/Math/Vector3.hpp"

namespace Exscitech
{
  class Mesh;
  class VolmapExscitech;
  class BulletUtility
  {

  public:

    typedef std::map<const std::string, btCollisionShape*> ShapeMap;
    typedef ShapeMap::const_iterator ShapeMapConstIter;

  public:

    static btCollisionShape*
    acquireBtCollisionShape (const std::string& shapeName);

    static bool
    addShapeToMap (const std::string& shapeName, btCollisionShape* shape,
        bool overrideIfFound = false);

    static bool
    createBulletConcaveTriMeshFromVolmap (const std::string& shapeName,
        VolmapExscitech* volmap, bool overrideIfFound = false);

    static bool
    createBulletConcaveTriMeshFromMeshInfo (const std::string& shapeName,
        Mesh* mesh, bool overrideIfFound);

    static bool
    createBulletConvexTriMeshFromMeshInfo (const std::string& shapeName,
        Mesh* mesh, bool overrideIfFound);

    static bool
    createBulletConvexHullFromMeshInfo (const std::string& shapeName,
        Mesh* mesh, bool overrideIfFound);

    static bool
    createBulletConcaveShapeFromVertices (const std::string& shapeName,
        const std::vector<Vector3f>& vertices, bool overrideIfFound = false);

    static bool
    createBulletMultiSphereShape (const std::string& shapeName,
        std::vector<Vector3f>& positions, std::vector<float>& radii,
        bool overrideIfFound = false);

    static btRigidBody*
    createRigidBody (const std::string& shapeName, float mass,
        const Vector3f& startingPosition, const Vector3f& initialVelocity);

    static btRigidBody*
    createRigidBody (btCollisionShape* shape, float mass,
        const Vector3f& startingPosition, const Vector3f& initialVelocity);

    static void
    createStaticSpheresFromPositions (btSphereShape* sphere,
        const std::vector<Vector3f>& positions, std::vector<btRigidBody*>& bodies);

    static void
    clearShapeMap ();

  private:

    BulletUtility ();

    static btIndexedMesh
    createBtIndexedMesh (std::vector<float>& vertices,
        std::vector<uint>& indices, int indexStride, int numTriangles,
        int vertexSizeInFloats, int numVertices, int vertexStride);

  private:

    static ShapeMap ms_shapeMap;
  };
}

#endif
