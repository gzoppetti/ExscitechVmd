/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Inform.h"
#include "utilities.h"

#define ISOSURFACE_INTERNAL 1
#include "Isosurface.h"
#include "VolumetricData.h"

IsoSurface::IsoSurface (void)
{
}

void
IsoSurface::clear (void)
{
  numtriangles = 0;
  v.clear ();
  n.clear ();
  f.clear ();
}

int
IsoSurface::compute (const VolumetricData *data, float isovalue, int step)
{
  int x, y, z;
  int tricount = 0;

  vol = data;

  // calculate cell axes
  vol->cell_axes (xax, yax, zax);
  vol->cell_dirs (xad, yad, zad);

  // flip normals if coordinate system is in the wrong handedness
  float vtmp[3];
  cross_prod (vtmp, xad, yad);
  if (dot_prod (vtmp, zad) < 0)
  {
    xad[0] *= -1;
    xad[1] *= -1;
    xad[2] *= -1;
    yad[0] *= -1;
    yad[1] *= -1;
    yad[2] *= -1;
    zad[0] *= -1;
    zad[1] *= -1;
    zad[2] *= -1;
  }

  for (z = 0; z < (vol->zsize - step); z += step)
  {
    for (y = 0; y < (vol->ysize - step); y += step)
    {
      for (x = 0; x < (vol->xsize - step); x += step)
      {
        int newTri = DoCell (x, y, z, isovalue, step);
        //fprintf (stderr, "New Tri for xyz (%d %d %d): %d\n", x, y, z, newTri);
        tricount += newTri;
      }
    }
  }

  return 1;
}

int
IsoSurface::DoCell (int x, int y, int z, float isovalue, int step)
{
  //fprintf (stderr, "x %d y %d z %d iso %f step %d\n", x, y, z, isovalue, step);
  GRIDCELL gc;
  int addr, row, plane, rowstep, planestep, tricount;
  TRIANGLE tris[5];

  row = vol->xsize;
  plane = vol->xsize * vol->ysize;

  addr = z * plane + y * row + x;
  rowstep = row * step;
  planestep = plane * step;
//  fprintf (stderr, "Row %d Plane %d addr %d rowstep %d planestep %d\n", row,
//      plane, addr, rowstep, planestep);
  gc.val[0] = vol->data[addr];
  gc.val[1] = vol->data[addr + step];
  gc.val[3] = vol->data[addr + rowstep];
  gc.val[2] = vol->data[addr + step + rowstep];
  gc.val[4] = vol->data[addr + planestep];
  gc.val[5] = vol->data[addr + step + planestep];
  gc.val[7] = vol->data[addr + rowstep + planestep];
  gc.val[6] = vol->data[addr + step + rowstep + planestep];

//  fprintf (stderr, "gc? %f %f %f %f %f %f %f %f\n", gc.val[0], gc.val[1],
//      gc.val[2], gc.val[3], gc.val[4], gc.val[5], gc.val[6], gc.val[7]);
  /*
   Determine the index into the edge table which
   tells us which vertices are inside of the surface
   */
  int cubeindex = 0;
  if (gc.val[0] < isovalue)
    cubeindex |= 1;
  if (gc.val[1] < isovalue)
    cubeindex |= 2;
  if (gc.val[2] < isovalue)
    cubeindex |= 4;
  if (gc.val[3] < isovalue)
    cubeindex |= 8;
  if (gc.val[4] < isovalue)
    cubeindex |= 16;
  if (gc.val[5] < isovalue)
    cubeindex |= 32;
  if (gc.val[6] < isovalue)
    cubeindex |= 64;
  if (gc.val[7] < isovalue)
    cubeindex |= 128;

  /* Cube is entirely in/out of the surface */
  if (edgeTable[cubeindex] == 0)
  {
    //fprintf(stderr, "Returning 0\n");
    return (0);
  }
  gc.cubeindex = cubeindex;

  gc.p[0].x = (float) x;
  gc.p[0].y = (float) y;
  gc.p[0].z = (float) z;

  VOXEL_GRADIENT_FAST(vol, x, y, z, &gc.g[0].x)

  gc.p[1].x = (float) x + step;
  gc.p[1].y = (float) y;
  gc.p[1].z = (float) z;

  VOXEL_GRADIENT_FAST(vol, x + step, y, z, &gc.g[1].x)

  gc.p[3].x = (float) x;
  gc.p[3].y = (float) y + step;
  gc.p[3].z = (float) z;

  VOXEL_GRADIENT_FAST(vol, x, y + step, z, &gc.g[3].x)

  gc.p[2].x = (float) x + step;
  gc.p[2].y = (float) y + step;
  gc.p[2].z = (float) z;

  VOXEL_GRADIENT_FAST(vol, x + step, y + step, z, &gc.g[2].x)

  gc.p[4].x = (float) x;
  gc.p[4].y = (float) y;
  gc.p[4].z = (float) z + step;

  VOXEL_GRADIENT_FAST(vol, x, y, z + step, &gc.g[4].x)

  gc.p[5].x = (float) x + step;
  gc.p[5].y = (float) y;
  gc.p[5].z = (float) z + step;

  VOXEL_GRADIENT_FAST(vol, x + step, y, z + step, &gc.g[5].x)

  gc.p[7].x = (float) x;
  gc.p[7].y = (float) y + step;
  gc.p[7].z = (float) z + step;

  VOXEL_GRADIENT_FAST(vol, x, y + step, z + step, &gc.g[7].x)

  gc.p[6].x = (float) x + step;
  gc.p[6].y = (float) y + step;
  gc.p[6].z = (float) z + step;

  VOXEL_GRADIENT_FAST(vol, x + step, y + step, z + step, &gc.g[6].x)

  // calculate vertices and facets for this cube,
  // calculate normals by interpolating between the negated 
  // normalized volume gradients for the 8 reference voxels
  tricount = Polygonise (gc, isovalue, (TRIANGLE *) &tris);

  if (tricount > 0)
  {
    int i;

    for (i = 0; i < tricount; i++)
    {
      float xx, yy, zz;
      float xn, yn, zn;

      f.append (numtriangles * 3);
      f.append (numtriangles * 3 + 1);
      f.append (numtriangles * 3 + 2);
      numtriangles++;

      // add new vertices and normals into vertex and normal lists
      xx = tris[i].p[0].x;
      yy = tris[i].p[0].y;
      zz = tris[i].p[0].z;

      v.append (
          (float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0]);
      v.append (
          (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1]);
      v.append (
          (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);

      xn = tris[i].n[0].x;
      yn = tris[i].n[0].y;
      zn = tris[i].n[0].z;
      n.append ((float) xn * xad[0] + yn * yad[0] + zn * zad[0]);
      n.append ((float) xn * xad[1] + yn * yad[1] + zn * zad[1]);
      n.append ((float) xn * xad[2] + yn * yad[2] + zn * zad[2]);

      xx = tris[i].p[1].x;
      yy = tris[i].p[1].y;
      zz = tris[i].p[1].z;
      v.append (
          (float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0]);
      v.append (
          (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1]);
      v.append (
          (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);
      xn = tris[i].n[1].x;
      yn = tris[i].n[1].y;
      zn = tris[i].n[1].z;
      n.append ((float) xn * xad[0] + yn * yad[0] + zn * zad[0]);
      n.append ((float) xn * xad[1] + yn * yad[1] + zn * zad[1]);
      n.append ((float) xn * xad[2] + yn * yad[2] + zn * zad[2]);

      xx = tris[i].p[2].x;
      yy = tris[i].p[2].y;
      zz = tris[i].p[2].z;
      v.append (
          (float) vol->origin[0] + xx * xax[0] + yy * yax[0] + zz * zax[0]);
      v.append (
          (float) vol->origin[1] + xx * xax[1] + yy * yax[1] + zz * zax[1]);
      v.append (
          (float) vol->origin[2] + xx * xax[2] + yy * yax[2] + zz * zax[2]);
      xn = tris[i].n[2].x;
      yn = tris[i].n[2].y;
      zn = tris[i].n[2].z;
      n.append ((float) xn * xad[0] + yn * yad[0] + zn * zad[0]);
      n.append ((float) xn * xad[1] + yn * yad[1] + zn * zad[1]);
      n.append ((float) xn * xad[2] + yn * yad[2] + zn * zad[2]);
    }
  }

  return tricount;
}

// normalize surface normals resulting from interpolation between 
// unnormalized volume gradients
void
IsoSurface::normalize ()
{
  int i;
  for (i = 0; i < n.num (); i += 3)
  {
    vec_normalize (&n[i]);
  }
}

// merge duplicated vertices detected by a simple windowed search
int
IsoSurface::vertexfusion (const VolumetricData *data, int offset, int len)
{
  int i, j, newverts, oldverts, faceverts, matchcount;

  faceverts = f.num ();
  oldverts = v.num () / 3;

  // abort if we get an empty list
  if (!faceverts || !oldverts)
    return 0;

  int * vmap = new int[oldverts];

  vmap[0] = 0;
  newverts = 1;
  matchcount = 0;

  for (i = 1; i < oldverts; i++)
  {
    int matchindex = -1;
    int start = ((newverts - offset) < 0) ? 0 : (newverts - offset);
    int end = ((start + len) > newverts) ? newverts : (start + len);
    int matched = 0;
    int vi = i * 3;
    for (j = start; j < end; j++)
    {
      int vj = j * 3;
      if (v[vi] == v[vj] && v[vi + 1] == v[vj + 1] && v[vi + 2] == v[vj + 2])
      {
        matched = 1;
        matchindex = j;
        matchcount++;
        break;
      }
    }

    if (matched)
    {
      vmap[i] = matchindex;
    }
    else
    {
      int vn = newverts * 3;
      v[vn] = v[vi];
      v[vn + 1] = v[vi + 1];
      v[vn + 2] = v[vi + 2];
      n[vn] = n[vi];
      n[vn + 1] = n[vi + 1];
      n[vn + 2] = n[vi + 2];
      vmap[i] = newverts;
      newverts++;
    }
  }

//  printf("Info) Vertex fusion: found %d shared vertices of %d, %d unique\n",
//         matchcount, oldverts, newverts);

  // zap the old face, vertex, and normal arrays and replace with the new ones
  for (i = 0; i < faceverts; i++)
  {
    f[i] = vmap[f[i]];
  }
  delete[] vmap;

  v.truncatelastn ((oldverts - newverts) * 3);
  n.truncatelastn ((oldverts - newverts) * 3);

  return 0;
}

/*
 Given a grid cell and an isolevel, calculate the triangular
 facets required to represent the isosurface through the cell.
 Return the number of triangular facets, the array "triangles"
 will be loaded up with the vertices at most 5 triangular facets.
 0 will be returned if the grid cell is either totally above
 of totally below the isolevel.
 This code calculates vertex normals by interpolating the volume gradients.
 */
int
IsoSurface::Polygonise (const GRIDCELL grid, const float isolevel,
    TRIANGLE *triangles)
{
  int i, ntriang;
  int cubeindex = grid.cubeindex;
  XYZ vertlist[12];
  XYZ normlist[12];

  /* Find the vertices where the surface intersects the cube */
  if (edgeTable[cubeindex] & 1)
    VertexInterp (isolevel, grid, 0, 1, &vertlist[0], &normlist[0]);
  if (edgeTable[cubeindex] & 2)
    VertexInterp (isolevel, grid, 1, 2, &vertlist[1], &normlist[1]);
  if (edgeTable[cubeindex] & 4)
    VertexInterp (isolevel, grid, 2, 3, &vertlist[2], &normlist[2]);
  if (edgeTable[cubeindex] & 8)
    VertexInterp (isolevel, grid, 3, 0, &vertlist[3], &normlist[3]);
  if (edgeTable[cubeindex] & 16)
    VertexInterp (isolevel, grid, 4, 5, &vertlist[4], &normlist[4]);
  if (edgeTable[cubeindex] & 32)
    VertexInterp (isolevel, grid, 5, 6, &vertlist[5], &normlist[5]);
  if (edgeTable[cubeindex] & 64)
    VertexInterp (isolevel, grid, 6, 7, &vertlist[6], &normlist[6]);
  if (edgeTable[cubeindex] & 128)
    VertexInterp (isolevel, grid, 7, 4, &vertlist[7], &normlist[7]);
  if (edgeTable[cubeindex] & 256)
    VertexInterp (isolevel, grid, 0, 4, &vertlist[8], &normlist[8]);
  if (edgeTable[cubeindex] & 512)
    VertexInterp (isolevel, grid, 1, 5, &vertlist[9], &normlist[9]);
  if (edgeTable[cubeindex] & 1024)
    VertexInterp (isolevel, grid, 2, 6, &vertlist[10], &normlist[10]);
  if (edgeTable[cubeindex] & 2048)
    VertexInterp (isolevel, grid, 3, 7, &vertlist[11], &normlist[11]);

  /* Create the triangle */
  ntriang = 0;
  for (i = 0; triTable[cubeindex][i] != -1; i += 3)
  {
    triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i]];
    triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i + 1]];
    triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i + 2]];
    triangles[ntriang].n[0] = normlist[triTable[cubeindex][i]];
    triangles[ntriang].n[1] = normlist[triTable[cubeindex][i + 1]];
    triangles[ntriang].n[2] = normlist[triTable[cubeindex][i + 2]];
    ntriang++;
  }

  return ntriang;
}

/*
 Linearly interpolate the position where an isosurface cuts
 an edge between two vertices, each with their own scalar value,
 interpolating vertex position and vertex normal based on the
 isovalue.
 */
void
IsoSurface::VertexInterp (float isolevel, const GRIDCELL grid, int ind1,
    int ind2, XYZ * vert, XYZ * norm)
{
  XYZ p1 = grid.p[ind1];
  XYZ p2 = grid.p[ind2];
  XYZ n1 = grid.g[ind1];
  XYZ n2 = grid.g[ind2];
  float valp1 = grid.val[ind1];
  float valp2 = grid.val[ind2];
  float isodiffp1 = isolevel - valp1;
  float diffvalp2p1 = valp2 - valp1;
  float mu = 0.0f;

  // if the difference between vertex values is zero or nearly
  // zero, we can get an IEEE NAN for mu.  We must either avoid this
  // by testing the denominator beforehand, by coping with the resulting
  // NAN value after the fact.  The only important thing is that mu be
  // assigned a value between zero and one.

#if 0
  if (fabsf(isodiffp1) < 0.00001)
  {
    *vert = p1;
    *norm = n1;
    return;
  }

  if (fabsf(isolevel-valp2) < 0.00001)
  {
    *vert = p2;
    *norm = n2;
    return;
  }

  if (fabsf(diffvalp2p1) < 0.00001)
  {
    *vert = p1;
    *norm = n1;
    return;
  }
#endif

  if (fabsf (diffvalp2p1) > 0.0f)
    mu = isodiffp1 / diffvalp2p1;

#if 0
  if (mu > 1.0f)
  mu=1.0f;

  if (mu < 0.0f)
  mu=0.0f;
#endif

  vert->x = p1.x + mu * (p2.x - p1.x);
  vert->y = p1.y + mu * (p2.y - p1.y);
  vert->z = p1.z + mu * (p2.z - p1.z);

  norm->x = n1.x + mu * (n2.x - n1.x);
  norm->y = n1.y + mu * (n2.y - n1.y);
  norm->z = n1.z + mu * (n2.z - n1.z);
}

