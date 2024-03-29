/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: DispCmds.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.85 $	$Date: 2009/04/29 15:42:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * DispCmds - different display commands which take data and put it in
 *	a storage space provided by a given VMDDisplayList object.
 *
 * Notes:
 *	1. All coordinates are stored as 3 points (x,y,z), even if meant
 * for a 2D object.  The 3rd coord for 2D objects will be ignored.
 ***************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//Exscitech
#include <GL/glew.h>
#include "Exscitech/Graphics/VolmapExscitech.hpp"

#ifdef VMDACTC
extern "C"
{
// XXX 
// The regular ACTC distribution compiles as plain C, need to send 
// a header file fix to Brad Grantham so C++ codes don't need this.
#include <tc.h>
}
#endif

#include "DispCmds.h"
#include "utilities.h"
#include "Matrix4.h"
#include "VMDDisplayList.h"
#include "Inform.h"
#include "VMDApp.h" // needed for texture serial numbers
//*************************************************************

// Pass a block o data to the command list.
#ifdef VMDCAVE
void DispCmdDataBlock::putdata(float *d, int n, VMDDisplayList *dobj)
{
  void *ptr = dobj->append(DDATABLOCK, n*sizeof(float));
  if (ptr == NULL)
  return;
  memcpy(ptr, d, n*sizeof(float));
}
#else
void
DispCmdDataBlock::putdata (float *d, int, VMDDisplayList *dobj)
{
  void *ptr = dobj->append (DDATABLOCK, sizeof(DispCmdDataBlock));
  if (ptr == NULL)
    return;
  data = d;
  memcpy (ptr, this, sizeof(DispCmdDataBlock));
}
#endif

//*************************************************************

// plot a point at the given position
void
DispCmdPoint::putdata (const float *newpos, VMDDisplayList *dobj)
{
  DispCmdPoint *ptr = (DispCmdPoint *) (dobj->append (DPOINT,
      sizeof(DispCmdPoint)));
  if (ptr == NULL)
    return;
  ptr->pos[0] = newpos[0];
  ptr->pos[1] = newpos[1];
  ptr->pos[2] = newpos[2];
}

//*************************************************************

// plot a sphere of specified radius at the given position
void
DispCmdSphere::putdata (float *newpos, float radius, VMDDisplayList *dobj)
{
  DispCmdSphere *ptr = (DispCmdSphere *) (dobj->append (DSPHERE,
      sizeof(DispCmdSphere)));
  if (ptr == NULL)
    return;
  ptr->pos_r[0] = newpos[0];
  ptr->pos_r[1] = newpos[1];
  ptr->pos_r[2] = newpos[2];
  ptr->pos_r[3] = radius;
}

void
DispCmdSphereIndex::putdata (int newpos, float newrad, VMDDisplayList *dobj)
{
  DispCmdSphereIndex *ptr = (DispCmdSphereIndex *) (dobj->append (DSPHERE_I,
      sizeof(DispCmdSphereIndex)));
  if (ptr == NULL)
    return;
  ptr->pos = newpos;
  ptr->rad = newrad;
}

void
DispCmdSphereArray::putdata (const float * spcenters, const float * spradii,
    const float * spcolors, int num_spheres, int sphere_res,
    VMDDisplayList * dobj)
{

  DispCmdSphereArray *ptr = (DispCmdSphereArray *) dobj->append (
      DSPHEREARRAY,
      sizeof(DispCmdSphereArray) + sizeof(float) * num_spheres * 3
          + sizeof(float) * num_spheres + sizeof(float) * num_spheres * 3
          + sizeof(int) * 2);
  if (ptr == NULL)
    return;
  ptr->numspheres = num_spheres;
  ptr->sphereres = sphere_res;

  float *centers;
  float *radii;
  float *colors;
  ptr->getpointers (centers, radii, colors);

  memcpy (centers, spcenters, sizeof(float) * num_spheres * 3);
  memcpy (radii, spradii, sizeof(float) * num_spheres);
  memcpy (colors, spcolors, sizeof(float) * num_spheres * 3);
}

//*************************************************************

void
DispCmdPointArray::putdata (const float * pcenters, const float * pcolors,
    float psize, int num_points, VMDDisplayList * dobj)
{

  DispCmdPointArray *ptr = (DispCmdPointArray *) dobj->append (
      DPOINTARRAY,
      sizeof(DispCmdPointArray) + sizeof(float) * num_points * 3
          + sizeof(float) * num_points * 3 + sizeof(float) + sizeof(int));
  if (ptr == NULL)
    return;
  ptr->size = psize;
  ptr->numpoints = num_points;

  float *centers;
  float *colors;
  ptr->getpointers (centers, colors);

  memcpy (centers, pcenters, sizeof(float) * num_points * 3);
  memcpy (colors, pcolors, sizeof(float) * num_points * 3);
}

//*************************************************************

void
DispCmdLitPointArray::putdata (const float * pcenters, const float * pnormals,
    const float * pcolors, float psize, int num_points, VMDDisplayList * dobj)
{

  DispCmdLitPointArray *ptr = (DispCmdLitPointArray *) dobj->append (
      DLITPOINTARRAY,
      sizeof(DispCmdLitPointArray) + sizeof(float) * num_points * 3
          + sizeof(float) * num_points * 3 + sizeof(float) * num_points * 3
          + sizeof(float) + sizeof(int));
  if (ptr == NULL)
    return;
  ptr->size = psize;
  ptr->numpoints = num_points;

  float *centers;
  float *normals;
  float *colors;
  ptr->getpointers (centers, normals, colors);

  memcpy (centers, pcenters, sizeof(float) * num_points * 3);
  memcpy (normals, pnormals, sizeof(float) * num_points * 3);
  memcpy (colors, pcolors, sizeof(float) * num_points * 3);
}

//*************************************************************

// plot a line at the given position
void
DispCmdLine::putdata (float *newpos1, float *newpos2, VMDDisplayList *dobj)
{
  DispCmdLine *ptr = (DispCmdLine *) (dobj->append (DLINE, sizeof(DispCmdLine)));
  if (ptr == NULL)
    return;
  memcpy (ptr->pos1, newpos1, 3 * sizeof(float));
  memcpy (ptr->pos2, newpos2, 3 * sizeof(float));
}

// draw a series of independent lines, (v0 v1), (v2 v3), (v4 v5)
void
DispCmdLineArray::putdata (float *v, int n, VMDDisplayList *dobj)
{
  void *ptr = dobj->append (DLINEARRAY, (1 + 6 * n) * sizeof(float));
  if (ptr == NULL)
    return;
  float *fltptr = (float *) ptr;
  *fltptr = (float) n;
  memcpy (fltptr + 1, v, 6 * n * sizeof(float));
}

// draw a series of connected polylines, (v0 v1 v2 v3 v4 v5)
void
DispCmdPolyLineArray::putdata (float *v, int n, VMDDisplayList *dobj)
{
  void *ptr = dobj->append (DPOLYLINEARRAY, (1 + 3 * n) * sizeof(float));
  if (ptr == NULL)
    return;
  float *fltptr = (float *) ptr;
  *fltptr = (float) n;
  memcpy (fltptr + 1, v, 3 * n * sizeof(float));
}

//*************************************************************
// draw a triangle

// set up the data for the DTRIANGLE drawing command
void
DispCmdTriangle::set_array (const float *p1, const float *p2, const float *p3,
    const float *n1, const float *n2, const float *n3, VMDDisplayList *dobj)
{
  DispCmdTriangle *ptr = (DispCmdTriangle *) (dobj->append (DTRIANGLE,
      sizeof(DispCmdTriangle)));
  if (ptr == NULL)
    return;
  memcpy (ptr->pos1, p1, 3 * sizeof(float));
  memcpy (ptr->pos2, p2, 3 * sizeof(float));
  memcpy (ptr->pos3, p3, 3 * sizeof(float));
  memcpy (ptr->norm1, n1, 3 * sizeof(float));
  memcpy (ptr->norm2, n2, 3 * sizeof(float));
  memcpy (ptr->norm3, n3, 3 * sizeof(float));
}

// put in new data, and put the command
void
DispCmdTriangle::putdata (const float *p1, const float *p2, const float *p3,
    VMDDisplayList *dobj)
{
  int i;
  float tmp1[3], tmp2[3], tmp3[3]; // precompute the normal for
  for (i = 0; i < 3; i++)
  { //   faster drawings later
    tmp1[i] = p2[i] - p1[i];
    tmp2[i] = p3[i] - p2[i];
  }
  cross_prod (tmp3, tmp1, tmp2);
  vec_normalize (tmp3);
  set_array (p1, p2, p3, tmp3, tmp3, tmp3, dobj);
}
void
DispCmdTriangle::putdata (const float *p1, const float *p2, const float *p3,
    const float *n1, const float *n2, const float *n3, VMDDisplayList *dobj)
{
  set_array (p1, p2, p3, n1, n2, n3, dobj);
}

//*************************************************************

// draw a square, given 3 of four points
void
DispCmdSquare::putdata (float *p1, float *p2, float *p3, VMDDisplayList *dobj)
{
  DispCmdSquare *ptr = (DispCmdSquare *) (dobj->append (DSQUARE,
      sizeof(DispCmdSquare)));
  if (ptr == NULL)
    return;
  int i;
  float tmp1[3], tmp2[3]; // precompute the normal for
  for (i = 0; i < 3; i++)
  { //   faster drawings later
    tmp1[i] = p2[i] - p1[i];
    tmp2[i] = p3[i] - p2[i];
  }
  cross_prod (ptr->norml, tmp1, tmp2);
  vec_normalize (ptr->norml);

  memcpy (ptr->pos1, p1, 3 * sizeof(float));
  memcpy (ptr->pos2, p2, 3 * sizeof(float));
  memcpy (ptr->pos3, p3, 3 * sizeof(float));
  for (i = 0; i < 3; i++)
    ptr->pos4[i] = p1[i] + tmp2[i]; // compute the fourth point
}

//*************************************************************

// draw a mesh consisting of vertices, facets, colors, normals etc.
void
DispCmdTriMesh::putdata (const float * vertices, const float * normals,
    const float * colors, int num_verts, const int * facets, int num_facets,
    int enablestrips, VMDDisplayList * dobj, DispCmdTriMesh* instance)
{
  int builtstrips = 0;

#if defined(VMDACTC) 
  if (enablestrips)
  {
    // Rearrange face data into triangle strips
    ACTCData *tc = actcNew();// intialize ACTC stripification library
    int fsize = num_facets * 3;
    int i, ind, ii;
    int iPrimCount = 0;
    int iCurrPrimSize;

    // XXX over-allocate the vertex and facet buffers to prevent an
    //     apparent bug in ACTC 1.1 from crashing VMD.  This was causing
    //     Surf surfaces to crash ACTC at times.
    int *p_iPrimSize = new int[fsize + 6];// num vertices in a primitive
    unsigned int *f2 = new uint[fsize + 6];

    if (tc == NULL)
    {
      msgErr << "ACTC initialization failed, using triangle mesh." << sendmsg;
    }
    else
    {
      msgInfo << "Performing ACTC Triangle Consolidation..." << sendmsg;

      // only produce strips, not fans, give a ridiculously high min value.
      actcParami(tc, ACTC_OUT_MIN_FAN_VERTS, 2147483647);

      // disabling honoring vertex winding order might allow ACTC to
      // consolidate more triangles into strips, but this is only useful
      // if VMD has two-sided lighting enabled.
      // actcParami(tc, ACTC_OUT_HONOR_WINDING, ACTC_TRUE);

      // send triangle data over to ACTC library
      actcBeginInput(tc);
      for (ii=0; ii < num_facets; ii++)
      {
        ind = ii * 3;
        if ((actcAddTriangle(tc, facets[ind], facets[ind + 1], facets[ind + 2])) != ACTC_NO_ERROR)
        {
          msgInfo << "ACTC Add Triangle Error." << sendmsg;
        }
      }
      actcEndInput(tc);

      // get triangle strips back from ACTC, loop through once to get sizes
      actcBeginOutput(tc);
      i = 0;
      while ((actcStartNextPrim(tc, &f2[i], &f2[i+1]) != ACTC_DATABASE_EMPTY))
      {
        iCurrPrimSize = 2; // if we're here, we got 2 vertices
        i+=2;// increment array position
        while (actcGetNextVert(tc, &f2[i]) != ACTC_PRIM_COMPLETE)
        {
          iCurrPrimSize++; // increment number of vertices for this primitive
          i++;// increment array position
        }

        p_iPrimSize[iPrimCount] = iCurrPrimSize; // save vertex count
        iPrimCount++;// increment primitive counter
      }
      actcEndOutput(tc);
      msgInfo << "ACTC: Created " << iPrimCount << " triangle strips" << sendmsg;
      msgInfo << "ACTC: Average vertices per strip = " << i / iPrimCount << sendmsg;

      // Draw triangle strips, uses double-sided lighting until we change
      // things to allow the callers to specify the desired lighting 
      // explicitly.
      DispCmdTriStrips::putdata(vertices, normals, colors, num_verts, p_iPrimSize, iPrimCount, f2, i, 1, dobj);

      // delete temporary memory
      delete [] f2;
      delete [] p_iPrimSize;

      // delete ACTC handle
      actcDelete(tc);

      builtstrips = 1;// don't generate a regular triangle mesh
    }
  }
#endif

  if (!builtstrips)
  {
    // make a triangle mesh (no strips)
    DispCmdTriMesh *ptr = (DispCmdTriMesh *) (dobj->append (
        DTRIMESH,
        sizeof(DispCmdTriMesh) + sizeof(float) * num_verts * 10
            + sizeof(int) * num_facets * 3));
    if (ptr == NULL)
      return;
    ptr->numverts = num_verts;
    ptr->numfacets = num_facets;
    float *cnv;
    int *f;
    ptr->getpointers (cnv, f);

    int i, ind, ind2;
    for (i = 0; i < num_verts; i++)
    {
      ind = i * 10;
      ind2 = i * 3;
      cnv[ind] = colors[ind2];
      cnv[ind + 1] = colors[ind2 + 1];
      cnv[ind + 2] = colors[ind2 + 2];
      cnv[ind + 3] = 1.0;
      cnv[ind + 4] = normals[ind2];
      cnv[ind + 5] = normals[ind2 + 1];
      cnv[ind + 6] = normals[ind2 + 2];
      cnv[ind + 7] = vertices[ind2];
      cnv[ind + 8] = vertices[ind2 + 1];
      cnv[ind + 9] = vertices[ind2 + 2];
    }
    memcpy (f, facets, ptr->numfacets * 3 * sizeof(int));

    if (instance != NULL)
    {
      Exscitech::VolmapExscitech::cache(cnv, f, num_verts, num_facets * 3);
      instance->numverts = num_verts;
      instance->numfacets = num_facets * 3;
      instance->m_vertexPointer = cnv;
      instance->m_indexPointer = f;
    }
  }
}

//*************************************************************

// draw a set of triangle strips
void
DispCmdTriStrips::putdata (const float * vertices, const float * normals,
    const float * colors, int num_verts, const int * verts_per_strip,
    int num_strips, const unsigned int * strip_data, const int num_strip_verts,
    int double_sided_lighting, VMDDisplayList * dobj)
{

  DispCmdTriStrips *ptr = (DispCmdTriStrips *) (dobj->append (
      DTRISTRIP,
      sizeof(DispCmdTriStrips) + sizeof(int *) * num_strips
          + sizeof(float) * num_verts * 10 + sizeof(int) * num_strip_verts
          + sizeof(int) * num_strips));
  if (ptr == NULL)
    return;
  ptr->numverts = num_verts;
  ptr->numstrips = num_strips;
  ptr->numstripverts = num_strip_verts;
  ptr->doublesided = double_sided_lighting;

  float *cnv;
  int *f;
  int *vertsperstrip;
  ptr->getpointers (cnv, f, vertsperstrip);

  // copy vertex,color,normal data
  int i, ind, ind2;
  for (i = 0; i < num_verts; i++)
  {
    ind = i * 10;
    ind2 = i * 3;
    cnv[ind] = colors[ind2];
    cnv[ind + 1] = colors[ind2 + 1];
    cnv[ind + 2] = colors[ind2 + 2];
    cnv[ind + 3] = 1.0;
    cnv[ind + 4] = normals[ind2];
    cnv[ind + 5] = normals[ind2 + 1];
    cnv[ind + 6] = normals[ind2 + 2];
    cnv[ind + 7] = vertices[ind2];
    cnv[ind + 8] = vertices[ind2 + 1];
    cnv[ind + 9] = vertices[ind2 + 2];
  }

  // copy vertices per strip data
  for (i = 0; i < num_strips; i++)
  {
    vertsperstrip[i] = verts_per_strip[i];
  }

  // copy face (triangle) data
  for (i = 0; i < num_strip_verts; i++)
  {
    f[i] = strip_data[i];
  }
}

//*************************************************************

void
DispCmdWireMesh::putdata (const float * vertices, const float * normals,
    const float * colors, int num_verts, const int * lines, int num_lines,
    VMDDisplayList * dobj)
{

  DispCmdWireMesh *ptr = (DispCmdWireMesh *) (dobj->append (
      DWIREMESH,
      sizeof(DispCmdWireMesh) + sizeof(float) * num_verts * 10
          + sizeof(int) * num_lines * 3));
  if (ptr == NULL)
    return;
  ptr->numverts = num_verts;
  ptr->numlines = num_lines;

  float *cnv;
  int *l;
  ptr->getpointers (cnv, l);

  int i, ind, ind2;
  for (i = 0; i < num_verts; i++)
  {
    ind = i * 10;
    ind2 = i * 3;
    cnv[ind] = colors[ind2];
    cnv[ind + 1] = colors[ind2 + 1];
    cnv[ind + 2] = colors[ind2 + 2];
    cnv[ind + 3] = 1.0;
    cnv[ind + 4] = normals[ind2];
    cnv[ind + 5] = normals[ind2 + 1];
    cnv[ind + 6] = normals[ind2 + 2];
    cnv[ind + 7] = vertices[ind2];
    cnv[ind + 8] = vertices[ind2 + 1];
    cnv[ind + 9] = vertices[ind2 + 2];
  }

  memcpy (l, lines, ptr->numlines * 2 * sizeof(int));
}

//*************************************************************
// plot a cylinder at the given position
// this is used to precalculate the cylinder data for speedup
// in renderers without hardware cylinders.  For example, the GL
// library.  There are res number of edges (with a norm, and two points)

DispCmdCylinder::DispCmdCylinder (void)
{
  lastres = 0;
}

void
DispCmdCylinder::putdata (const float *pos1, const float *pos2, float rad,
    int res, int filled, VMDDisplayList *dobj)
{

  float lenaxis[3];
  vec_sub (lenaxis, pos1, pos2); // check that it's valid
  if (dot_prod (lenaxis, lenaxis) == 0.0 || res <= 0)
    return;

  if (lastres != res)
  {
    rot[0] = cos ((float) VMD_TWOPI / (float) res);
    rot[1] = sin ((float) VMD_TWOPI / (float) res);
  }
  lastres = res;
  size_t size = (9 + res * 3 * 3) * sizeof(float);

  float *pos = (float *) (dobj->append (DCYLINDER, size));
  if (pos == NULL)
    return;

  memcpy (pos, pos1, 3 * sizeof(float));
  memcpy (pos + 3, pos2, 3 * sizeof(float));
  pos[6] = rad;
  pos[7] = (float) res;
  pos[8] = (float) filled;

  float axis[3];
  vec_sub (axis, pos1, pos2);
  vec_normalize (axis);
  int i; // find an axis not aligned with the cylinder
  if (fabs (axis[0]) < fabs (axis[1]) && fabs (axis[0]) < fabs (axis[2]))
  {
    i = 0;
  }
  else if (fabs (axis[1]) < fabs (axis[2]))
  {
    i = 1;
  }
  else
  {
    i = 2;
  }
  float perp[3];
  perp[i] = 0; // this is not aligned with the cylinder
  perp[(i + 1) % 3] = axis[(i + 2) % 3];
  perp[(i + 2) % 3] = -axis[(i + 1) % 3];
  vec_normalize (perp);
  float perp2[3];
  cross_prod (perp2, axis, perp); // find a normal to the cylinder

  float *posptr = pos + 9;
  float m = rot[0], n = rot[1];
  for (int h = 0; h < res; h++)
  {
    float tmp0, tmp1, tmp2;

    tmp0 = m * perp[0] + n * perp2[0]; // add the normal
    tmp1 = m * perp[1] + n * perp2[1];
    tmp2 = m * perp[2] + n * perp2[2];

    posptr[0] = tmp0; // add the normal
    posptr[1] = tmp1;
    posptr[2] = tmp2;

    posptr[3] = pos2[0] + rad * tmp0; // start
    posptr[4] = pos2[1] + rad * tmp1;
    posptr[5] = pos2[2] + rad * tmp2;

    posptr[6] = posptr[3] + lenaxis[0]; // and end of the edge
    posptr[7] = posptr[4] + lenaxis[1];
    posptr[8] = posptr[5] + lenaxis[2];
    posptr += 9;
    // use angle addition formulae:
    // cos(A+B) = cos A cos B - sin A sin B
    // sin(A+B) = cos A sin B + sin A cos B
    float mtmp = rot[0] * m - rot[1] * n;
    float ntmp = rot[0] * n + rot[1] * m;
    m = mtmp;
    n = ntmp;
  }
}

//*************************************************************

void
DispCmdCone::putdata (float *p1, float *p2, float newrad, float newrad2,
    int newres, VMDDisplayList *dobj)
{
  DispCmdCone *ptr = (DispCmdCone *) (dobj->append (DCONE, sizeof(DispCmdCone)));
  if (ptr == NULL)
    return;
  memcpy (ptr->pos1, p1, 3 * sizeof(float));
  memcpy (ptr->pos2, p2, 3 * sizeof(float));
  ptr->radius = newrad;
  ptr->radius2 = newrad2;
  ptr->res = newres;
}

// put in new data, and put the command
void
DispCmdColorIndex::putdata (int newcol, VMDDisplayList *dobj)
{
  DispCmdColorIndex *ptr = (DispCmdColorIndex *) (dobj->append (DCOLORINDEX,
      sizeof(DispCmdColorIndex)));
  if (ptr == NULL)
    return;
  ptr->color = newcol;
}

//*************************************************************

// display text at the given text coordinates
void
DispCmdText::putdata (const float *c, const char *s, VMDDisplayList *dobj)
{
  if (s != NULL)
  {
    size_t len = strlen (s) + 1;
    char *buf = (char *) (dobj->append (DTEXT, len + 3 * sizeof(float)));
    if (buf == NULL)
      return;
    ((float *) buf)[0] = c[0];
    ((float *) buf)[1] = c[1];
    ((float *) buf)[2] = c[2];
    memcpy (buf + 3 * sizeof(float), s, len);
  }
}

void
DispCmdTextOffset::putdata (float ox, float oy, VMDDisplayList *dobj)
{
  DispCmdTextOffset *cmd = (DispCmdTextOffset *) (dobj->append (DTEXTOFFSET,
      sizeof(DispCmdTextOffset)));
  cmd->x = ox;
  cmd->y = oy;
}

//*************************************************************
// include comments in the display list, useful for Token Rendering
void
DispCmdComment::putdata (const char *newtxt, VMDDisplayList *dobj)
{
  char *buf = (char *) dobj->append (DCOMMENT, strlen (newtxt) + 1);
  if (buf == NULL)
    return;
  memcpy (buf, newtxt, strlen (newtxt) + 1);
}

//*************************************************************

void
DispCmdTextSize::putdata (float size1, VMDDisplayList *dobj)
{
  DispCmdTextSize *ptr = (DispCmdTextSize *) dobj->append (DTEXTSIZE,
      sizeof(DispCmdTextSize));
  if (ptr == NULL)
    return;
  ptr->size = size1;
}

//*************************************************************

void
DispCmdVolSlice::putdata (int mode, const float *pnormal, const float *verts,
    const float *texs, VMDDisplayList *dobj)
{

  DispCmdVolSlice *cmd = (DispCmdVolSlice *) dobj->append (DVOLSLICE,
      sizeof(DispCmdVolSlice));
  if (cmd == NULL)
    return;

  cmd->texmode = mode;
  memcpy (cmd->normal, pnormal, 3 * sizeof(float));
  memcpy (cmd->v, verts, 12 * sizeof(float));
  memcpy (cmd->t, texs, 12 * sizeof(float));
}

//*************************************************************

void
DispCmdVolumeTexture::putdata (unsigned long texID, const int size[3],
    unsigned char *texptr, const float pv0[3], const float pv1[3],
    const float pv2[3], const float pv3[3], VMDDisplayList *dobj)
{

  DispCmdVolumeTexture *cmd = (DispCmdVolumeTexture *) dobj->append (
      DVOLUMETEXTURE, sizeof(DispCmdVolumeTexture));

  if (cmd == NULL)
    return;

  cmd->ID = texID;
  cmd->xsize = size[0];
  cmd->ysize = size[1];
  cmd->zsize = size[2];
  cmd->texmap = texptr;
  memcpy (cmd->v0, pv0, 3 * sizeof(float));
  memcpy (cmd->v1, pv1, 3 * sizeof(float));
  memcpy (cmd->v2, pv2, 3 * sizeof(float));
  memcpy (cmd->v3, pv3, 3 * sizeof(float));
}

//*************************************************************
// put in new data, and put the command
void
DispCmdSphereRes::putdata (int newres, VMDDisplayList *dobj)
{
  DispCmdSphereRes *ptr = (DispCmdSphereRes *) dobj->append (DSPHERERES,
      sizeof(DispCmdSphereRes));
  if (ptr == NULL)
    return;
  ptr->res = newres;
}

//*************************************************************

// put in new data, and put the command
void
DispCmdSphereType::putdata (int newtype, VMDDisplayList *dobj)
{
  DispCmdSphereType *ptr = (DispCmdSphereType *) dobj->append (DSPHERETYPE,
      sizeof(DispCmdSphereType));
  if (ptr == NULL)
    return;
  ptr->type = newtype;
}

//*************************************************************

// put in new data, and put the command
void
DispCmdLineType::putdata (int newtype, VMDDisplayList *dobj)
{
  DispCmdLineType* ptr = (DispCmdLineType *) dobj->append (DLINESTYLE,
      sizeof(DispCmdLineType));
  if (ptr == NULL)
    return;
  ptr->type = newtype;
}

//*************************************************************

void
DispCmdLineWidth::putdata (int newwidth, VMDDisplayList *dobj)
{
  DispCmdLineWidth * ptr = (DispCmdLineWidth *) dobj->append (DLINEWIDTH,
      sizeof(DispCmdLineWidth));
  if (ptr == NULL)
    return;
  ptr->width = newwidth;
}

//*************************************************************

void
DispCmdPickPoint::putdata (float *pos, int newtag, VMDDisplayList *dobj)
{
  DispCmdPickPoint *ptr = (DispCmdPickPoint *) (dobj->append (DPICKPOINT,
      sizeof(DispCmdPickPoint)));
  if (ptr == NULL)
    return;
  memcpy (ptr->postag, pos, 3 * sizeof(float));
  ptr->tag = newtag;
}

//*************************************************************

// put in new data, and put the command
void
DispCmdPickPointIndex::putdata (int newpos, int newtag, VMDDisplayList *dobj)
{
  DispCmdPickPointIndex *ptr = (DispCmdPickPointIndex *) (dobj->append (
      DPICKPOINT_I, sizeof(DispCmdPickPointIndex)));
  if (ptr == NULL)
    return;
  ptr->pos = newpos;
  ptr->tag = newtag;
}

//*************************************************************

// put in new data, and put the command
void
DispCmdPickPointIndexArray::putdata (int num, int numsel, int *onoff,
    VMDDisplayList *dobj)
{
  if (numsel < 1)
    return;

  int i;
  DispCmdPickPointIndexArray *ptr;

  if (num == numsel)
  {
    // if all indices in a contiguous block are enabled (e.g. "all" selection)
    // then there's no need to actually store the pick point indices
    ptr = (DispCmdPickPointIndexArray *) (dobj->append (DPICKPOINT_IARRAY,
        sizeof(DispCmdPickPointIndexArray)));
  }
  else
  {
    // if only some of the indices are selected, then we allocate storage
    // for the list of indices to be copied in.
    ptr = (DispCmdPickPointIndexArray *) (dobj->append (DPICKPOINT_IARRAY,
        sizeof(DispCmdPickPointIndexArray) + sizeof(int) * numsel));
  }

  if (ptr == NULL)
    return;

  ptr->numpicks = numsel;

  if (num == numsel)
  {
    // if all indices are selected note it, and we're done.
    ptr->allselected = 1;
  }
  else
  {
    // if only some indices are selected, copy in the selected ones
    ptr->allselected = 0;
    int *tags;
    ptr->getpointers (tags);

    // copy tags for selected/enabled indices
    int cp = 0;
    for (i = 0; i < num; i++)
    {
      if (onoff[i])
      {
        tags[cp] = i;
        cp++;
      }
    }
  }
}

//*************************************************************

// put in new data, and put the command
void
DispCmdPickPointIndexArray::putdata (int num, int *indices,
    VMDDisplayList *dobj)
{
  DispCmdPickPointIndexArray *ptr;

  ptr = (DispCmdPickPointIndexArray *) (dobj->append (DPICKPOINT_IARRAY,
      sizeof(DispCmdPickPointIndexArray) + sizeof(int) * num));

  if (ptr == NULL)
    return;

  ptr->numpicks = num;
  ptr->allselected = 0; // use the index array entries
  int *tags;
  ptr->getpointers (tags);
  memcpy (tags, indices, num * sizeof(int));
}

