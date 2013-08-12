#ifndef EXSCITECHCONVEXDECOMP
#define EXSCITECHCONVEXDECOMP

#include <vector>

#include <bullet/ConvexDecomposition/ConvexDecomposition.h>
#include <bullet/BulletCollision/CollisionShapes/btConvexHullShape.h>
#include <bullet/LinearMath/btGeometryUtil.h>
class ExscitechConvexDecomp : public ConvexDecomposition::ConvexDecompInterface
{
public:

  std::vector<btConvexHullShape*> m_convexShapes;
  std::vector<btVector3> m_convexCentroids;
  std::vector<btTriangleMesh*> m_triMeshes;
  std::vector<btCollisionShape*> m_collisionShapes;

  int mBaseCount;
  int mHullCount;

  ExscitechConvexDecomp () :
      mBaseCount (0), mHullCount (0)
  {
  }

  virtual void
  ConvexDecompResult (ConvexDecomposition::ConvexResult &result)
  {

    btTriangleMesh* trimesh = new btTriangleMesh ();
    m_triMeshes.push_back (trimesh);

    //btVector3 localScaling (6.f, 6.f, 6.f);

    //calc centroid, to shift vertices around center of mass
    btVector3 centroid;
    centroid.setValue (0, 0, 0);

    btAlignedObjectArray<btVector3> vertices;

    if (1)
    {

      for (unsigned int i = 0; i < result.mHullVcount; i++)
      {
        btVector3 vertex (result.mHullVertices[i * 3],
            result.mHullVertices[i * 3 + 1], result.mHullVertices[i * 3 + 2]);
        //vertex *= localScaling;
        centroid += vertex;

      }
    }

    centroid *= 1.f / (float (result.mHullVcount));

    if (1)
    {
      for (unsigned int i = 0; i < result.mHullVcount; i++)
      {
        btVector3 vertex (result.mHullVertices[i * 3],
            result.mHullVertices[i * 3 + 1], result.mHullVertices[i * 3 + 2]);
        //vertex *= localScaling;
        vertex -= centroid;
        vertices.push_back (vertex);
      }
    }

    if (1)
    {
      const unsigned int *src = result.mHullIndices;
      for (unsigned int i = 0; i < result.mHullTcount; i++)
      {
        unsigned int index0 = *src++;
        unsigned int index1 = *src++;
        unsigned int index2 = *src++;

        btVector3 vertex0 (result.mHullVertices[index0 * 3],
            result.mHullVertices[index0 * 3 + 1],
            result.mHullVertices[index0 * 3 + 2]);
        btVector3 vertex1 (result.mHullVertices[index1 * 3],
            result.mHullVertices[index1 * 3 + 1],
            result.mHullVertices[index1 * 3 + 2]);
        btVector3 vertex2 (result.mHullVertices[index2 * 3],
            result.mHullVertices[index2 * 3 + 1],
            result.mHullVertices[index2 * 3 + 2]);
        //vertex0 *= localScaling;
        //vertex1 *= localScaling;
        //vertex2 *= localScaling;

        vertex0 -= centroid;
        vertex1 -= centroid;
        vertex2 -= centroid;

        trimesh->addTriangle (vertex0, vertex1, vertex2);

        index0 += mBaseCount;
        index1 += mBaseCount;
        index2 += mBaseCount;
      }
    }

    float collisionMargin = 0.01f;

    btAlignedObjectArray<btVector3> planeEquations;
    btGeometryUtil::getPlaneEquationsFromVertices (vertices, planeEquations);

    btAlignedObjectArray<btVector3> shiftedPlaneEquations;
    for (int p = 0; p < planeEquations.size (); p++)
    {
      btVector3 plane = planeEquations[p];
      plane[3] += collisionMargin;
      shiftedPlaneEquations.push_back (plane);
    }
    btAlignedObjectArray<btVector3> shiftedVertices;
    btGeometryUtil::getVerticesFromPlaneEquations (shiftedPlaneEquations,
        shiftedVertices);

    btConvexHullShape* convexShape = new btConvexHullShape (
        &(shiftedVertices[0].getX ()), shiftedVertices.size ());

//    btConvexHullShape* convexShape = new btConvexHullShape (
//        &(vertices[0].getX ()), vertices.size ());

    convexShape->setMargin (0.01f);
    m_convexShapes.push_back (convexShape);
    m_convexCentroids.push_back (centroid);
    m_collisionShapes.push_back (convexShape);
    mBaseCount += result.mHullVcount; // advance the 'base index' counter.

  }
};

#endif
