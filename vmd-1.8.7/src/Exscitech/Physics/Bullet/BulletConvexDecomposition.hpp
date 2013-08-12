#ifndef BULLETCONVEXDECOMPOSITION_H_
#define BULLETCONVEXDECOMPOSITION_H_

#include <bullet/HACD/hacdHACD.h>
#include <bullet/HACD/hacdVector.h>
#include <bullet/ConvexDecomposition/ConvexBuilder.h>

#include "Exscitech/Physics/Bullet/ExscitechConvexDecomp.hpp"

class BulletConvexDecomposition
{
public:

  void
  decomp (float* vertices, int* indices, int numVertices, int numIndices)
  {
    ConvexDecomposition::DecompDesc desc;
    desc.mVcount = numVertices;
    desc.mVertices = vertices;
    desc.mTcount = numIndices / 3;
    desc.mIndices = (unsigned int*) indices;
    desc.mDepth = 5;
    desc.mCpercent = 40;
    desc.mPpercent = 20;
    desc.mMaxVertices = 32;
    desc.mSkinWidth = 0.1f;

    ExscitechConvexDecomp cd;
    desc.mCallback = &cd;
    ConvexBuilder cb (desc.mCallback);
    cb.process (desc);

    // D-D-Did we do it?
    /*
     *   std::vector<btConvexHullShape*> m_convexShapes;
     std::vector<btVector3> m_convexCentroids;
     std::vector<btTriangleMesh*> m_triMeshes;
     std::vector<btCollisionShape*> m_collisionShapes;

     int mBaseCount;
     int mHullCount;
     */
    int baseCount = cd.mBaseCount;
    int hullCount = cd.mHullCount;
    int numShapes = cd.m_convexShapes.size ();
    int numTriMesh = cd.m_triMeshes.size ();

    fprintf (stderr, "Base %d Hull %d shape %d tri %d\n", baseCount, hullCount,
        numShapes, numTriMesh);

    centerPositions = cd.m_convexCentroids;
    hulls = cd.m_convexShapes;

    for (unsigned int i = 0; i < cd.m_triMeshes.size (); ++i)
    {
      delete cd.m_triMeshes[i];
    }

//    for(unsigned int i = 0; i < cd.m_collisionShapes.size(); ++i)
//    {
//      delete cd.m_collisionShapes[i];
//    }
  }

  // We need to split it up :(
  void
  decomp2 (float* vertices, int* indices, unsigned int numVertices, unsigned int numIndices)
  {
    static const unsigned int numSections = 10;

    std::vector<HACD::Vec3<HACD::Real> > points;

    for (unsigned int i = 0; i < numVertices; i++)
    {
      int index = i * 3;
      HACD::Vec3<HACD::Real> vertex (vertices[index], vertices[index + 1],
          vertices[index + 2]);
      points.push_back (vertex);
    }
    // 303 indices
    // 101 in section at first
    // 101 % 3 = 2
    // 3 - 2 = 1
    // numVertsInSection = 102;
    // 0-101 102 - 203 204 -302
    // 102 102 99

    unsigned int numVertsInSection = numIndices / numSections;
    numVertsInSection +=  3 - numVertsInSection % 3;

    for (unsigned int section = 0; section < numSections; ++section)
    {
      std::vector<HACD::Vec3<long> > triangles;

      unsigned int start = section * numVertsInSection;
      unsigned int end = start + numVertsInSection;
      if (end > numIndices)
        end = numIndices;

      for (unsigned int i = start; i < end; i += 3)
      {
        HACD::Vec3<long> triangle (indices[i], indices[i + 1], indices[i + 2]);
        triangles.push_back (triangle);
      }

      HACD::HACD myHACD;
      myHACD.SetPoints (&points[0]);
      myHACD.SetNPoints (points.size ());
      myHACD.SetTriangles (&triangles[0]);
      myHACD.SetNTriangles (triangles.size ());
      myHACD.SetCompacityWeight (0.1);
      myHACD.SetVolumeWeight (0.0);

      // HACD parameters
      size_t nClusters = 4;
      double concavity = 100;
      bool addExtraDistPoints = true;
      bool addNeighboursDistPoints = false;
      bool addFacesPoints = false;

      myHACD.SetNClusters (nClusters); // minimum number of clusters
      myHACD.SetNVerticesPerCH (32); // max of 100 vertices per convex-hull
      myHACD.SetConcavity (concavity); // maximum concavity
      myHACD.SetAddExtraDistPoints (addExtraDistPoints);
      myHACD.SetAddNeighboursDistPoints (addNeighboursDistPoints);
      myHACD.SetAddFacesPoints (addFacesPoints);

      myHACD.Compute ();
      nClusters = myHACD.GetNClusters ();
      fprintf (stderr, "Num clusters after compute: %lu\n", nClusters);

      btTransform trans;
      trans.setIdentity ();

      ExscitechConvexDecomp cd;

      for (unsigned int c = 0; c < nClusters; c++)
      {
        //generate convex result
        size_t nPoints = myHACD.GetNPointsCH (c);
        size_t nTriangles = myHACD.GetNTrianglesCH (c);

        float* vertices = new float[nPoints * 3];
        unsigned int* triangles = new unsigned int[nTriangles * 3];

        HACD::Vec3<HACD::Real> *pointsCH = new HACD::Vec3<HACD::Real>[nPoints];
        HACD::Vec3<long> * trianglesCH = new HACD::Vec3<long>[nTriangles];
        myHACD.GetCH (c, pointsCH, trianglesCH);

        // points
        for (size_t v = 0; v < nPoints; v++)
        {
          vertices[3 * v] = pointsCH[v].X ();
          vertices[3 * v + 1] = pointsCH[v].Y ();
          vertices[3 * v + 2] = pointsCH[v].Z ();
        }
        // triangles
        for (size_t f = 0; f < nTriangles; f++)
        {
          triangles[3 * f] = trianglesCH[f].X ();
          triangles[3 * f + 1] = trianglesCH[f].Y ();
          triangles[3 * f + 2] = trianglesCH[f].Z ();
        }

        delete[] pointsCH;
        delete[] trianglesCH;

        ConvexResult r (nPoints, vertices, nTriangles, triangles);
        cd.ConvexDecompResult (r);
      }

      for (unsigned int i = 0; i < cd.m_convexShapes.size (); ++i)
      {
        hulls.push_back (cd.m_convexShapes[i]);
      }
      for (unsigned int i = 0; i < cd.m_convexCentroids.size (); ++i)
      {
        centerPositions.push_back (cd.m_convexCentroids[i]);
      }
//    hulls = cd.m_convexShapes;
//    centerPositions = cd.m_convexCentroids;

      for (unsigned int i = 0; i < cd.m_triMeshes.size (); ++i)
      {
        delete cd.m_triMeshes[i];
      }

//      for(unsigned int i = 0; i < cd.m_collisionShapes.size(); ++i)
//      {
//        delete cd.m_collisionShapes[i];
//      }
    }
  }

  std::vector<btConvexHullShape*> hulls;
  std::vector<btVector3> centerPositions;
};

#endif
