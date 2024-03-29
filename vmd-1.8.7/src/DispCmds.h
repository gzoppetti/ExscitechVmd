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
 *	$RCSfile: DispCmds.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.89 $	$Date: 2009/04/29 15:42:58 $
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
#ifndef DISPCMDS_H
#define DISPCMDS_H

class VMDDisplayList;

/// enum for all display commands
/// draw commands with a _I suffix use indices into a DATA block, not the
/// coordinates themselves
enum
{
  DPOINT,
  DPOINTARRAY,
  DLITPOINTARRAY,
  DLINE,
  DLINEARRAY,
  DPOLYLINEARRAY,
  DCYLINDER,
  DSPHERE,
  DSPHERE_I,
  DSPHEREARRAY,
  DTRIANGLE,
  DSQUARE,
  DCONE,
  DTRIMESH,
  DTRISTRIP,
  DWIREMESH,
  DCOLORINDEX,
  DMATERIALON,
  DMATERIALOFF,
  DTEXT,
  DTEXTSIZE,
  DCOMMENT,
  DTEXTOFFSET,
  DCLIPPLANE,
  DVOLSLICE,
  DVOLTEXON,
  DVOLTEXOFF,
  DVOLUMETEXTURE,
  DSPHERERES,
  DSPHERETYPE,
  DLINEWIDTH,
  DLINESTYLE,
  DPICKPOINT,
  DPICKPOINT_I,
  DPICKPOINT_IARRAY,
  DDATABLOCK,
  DLASTCOMMAND = -1
};

/// enum with different sphere and line types
enum
{
  SOLIDSPHERE, POINTSPHERE
};
enum
{
  SOLIDLINE, DASHEDLINE
};

/// Pass a block of data to the command list.  If we're running in the
/// CAVE, the whole block needs to be copied; otherwise, we save space
/// by simply passing the pointer.
struct DispCmdDataBlock
{
  void
  putdata (float *, int, VMDDisplayList *); // 2nd argument is # floats
  float *data;
};

/// plot a point at the given position
struct DispCmdPoint
{
  void
  putdata (const float *, VMDDisplayList *);
  float pos[3];
};

/// draw a sphere of specified radius at the given position
struct DispCmdSphere
{
  void
  putdata (float *, float, VMDDisplayList *);
  float pos_r[4]; ///< position and radius together, for backward compatibility 
                  ///< and efficiency
};

/// draw a sphere of specified radius at the given position, using indices
struct DispCmdSphereIndex
{
  void
  putdata (int, float, VMDDisplayList *);
  int pos;
  float rad;
};

/// draw spheres from the specified position, radii, and color arrays
struct DispCmdSphereArray
{
  static void
  putdata (const float * centers, const float * radii, const float * colors,
      int num_spheres, int sphere_res, VMDDisplayList * dobj);

  inline void
  getpointers (float *& centers, float *& radii, float *& colors) const
  {
    char *rawptr = (char *) this;
    centers = (float *) (rawptr + sizeof(DispCmdSphereArray));
    radii = (float *) (rawptr + sizeof(DispCmdSphereArray)
        + sizeof(float) * numspheres * 3);
    colors = (float *) (rawptr + sizeof(DispCmdSphereArray)
        + sizeof(float) * numspheres * 3 + sizeof(float) * numspheres);
  }

  int numspheres;
  int sphereres;
};

/// draw points from the specified position, and color arrays, with given size
struct DispCmdPointArray
{
  static void
  putdata (const float * centers, const float * colors, float size,
      int num_points, VMDDisplayList * dobj);

  inline void
  getpointers (float *& centers, float *& colors) const
  {
    char *rawptr = (char *) this;
    centers = (float *) (rawptr + sizeof(DispCmdPointArray));
    colors = (float *) (rawptr + sizeof(DispCmdPointArray)
        + sizeof(float) * numpoints * 3);
  }
  float size;
  int numpoints;
};

/// draw points from the specified position, and color arrays, with given size
struct DispCmdLitPointArray
{
  static void
  putdata (const float * centers, const float * normals, const float * colors,
      float size, int num_points, VMDDisplayList * dobj);

  inline void
  getpointers (float *& centers, float *& normals, float *& colors) const
  {
    char *rawptr = (char *) this;
    centers = (float *) (rawptr + sizeof(DispCmdLitPointArray));
    normals = (float *) (rawptr + sizeof(DispCmdLitPointArray)
        + sizeof(float) * numpoints * 3);
    colors = (float *) (rawptr + sizeof(DispCmdLitPointArray)
        + sizeof(float) * numpoints * 6);
  }

  float size;
  int numpoints;
};

/// plot a line at the given position
struct DispCmdLine
{
  void
  putdata (float *, float *, VMDDisplayList *);
  float pos1[3];
  float pos2[3];
};

/// plot a series of lines, all with the same color
/// the array should be of the form v1 v2 v1 v2 ...
/// Note that the data is not stored in the struct; we copy directly into
/// the display list.  That way you don't have to keep this struct around
/// after calling putdata.   Kinda like DispCmdDataBlock.
struct DispCmdLineArray
{
  void
  putdata (float *v, int nlines, VMDDisplayList *);
};

/// plot a series of connected polylines, all with the same color
/// the array should be of the form v1 v2 v3 v4 ...
/// Note that the data is not stored in the struct; we copy directly into
/// the display list.  That way you don't have to keep this struct around
/// after calling putdata.   Kinda like DispCmdDataBlock.
struct DispCmdPolyLineArray
{
  void
  putdata (float *v, int nlines, VMDDisplayList *);
};

/// draw a triangle, given the three points (computes the normals
///  from the cross product) and all normals are the same
///   -or-
/// draw the triangle, given the three points and the three normals
struct DispCmdTriangle
{
  void
  putdata (const float *, const float *, const float *, VMDDisplayList *);
  void
  putdata (const float *, const float *, const float *, const float *,
      const float *, const float *, VMDDisplayList *);

  float pos1[3], pos2[3], pos3[3];
  float norm1[3], norm2[3], norm3[3];
  void
  set_array (const float *, const float *, const float *, const float *,
      const float *, const float *, VMDDisplayList *);
};

/// draw a square, given 3 of four points
struct DispCmdSquare
{
  float pos1[3], pos2[3], pos3[3], pos4[3];
  float norml[3];
  void
  putdata (float *p1, float *p2, float *p3, VMDDisplayList *);
};

/// draw a mesh consisting of vertices, facets, colors, normals etc.
struct DispCmdTriMesh
{
  static void
  putdata (const float * vertices, const float * normals, const float * colors,
      int num_verts, const int * facets, int num_facets, int enablestrips,
      VMDDisplayList *, DispCmdTriMesh* instance = NULL);

  /// retrieve pointers to data following DispCmd in the display list
  /// float * cnv : array of colors, normals, vertices
  /// int * f     : facet vertex index array

  inline void
  getpointers (float *& cnv, int *& f) const
  {
    cnv = (float *) (((char *) this) + sizeof(DispCmdTriMesh));
    f = (int *) (cnv + 10 * numverts);
  }

  void
  getCachedInformation (float*& vertices, int*& indices, int& numVertices,
      int& numIndices) const
  {
    vertices = m_vertexPointer;
    indices = m_indexPointer;
    numVertices = numverts;
    numIndices = numfacets;
  }
  int numverts; ///< number of vertices in mesh
  int numfacets; ///< number of facets
  float* m_vertexPointer;
  int* m_indexPointer;
};

/// draw a set of triangle strips
struct DispCmdTriStrips
{
  static void
  putdata (const float * vertices, const float * normals, const float * colors,
      int num_verts, const int * verts_per_strip, int num_strips,
      const unsigned int * strip_data, const int num_strip_verts,
      int double_sided_lighting, VMDDisplayList * dobj);

  /// float * cnv;          array of colors, normals, vertices
  /// int * f;              facet vertex index array
  /// int * vertsperstrip;  array of vertex count per strip index
  inline void
  getpointers (float *& cnv, int *& f, int *& vertsperstrip) const
  {

    char *rawptr = (char *) this;
    cnv = (float *) (rawptr + sizeof(DispCmdTriStrips));

    f = (int *) (rawptr + sizeof(DispCmdTriStrips)
        + sizeof(float) * numverts * 10);

    vertsperstrip = (int *) (rawptr + sizeof(DispCmdTriStrips)
        + sizeof(float) * numverts * 10 + sizeof(int) * numstripverts);
  }

  int numverts; ///< number of vertices in mesh
  int numstrips; ///< total number of strips
  int numstripverts; ///< total number of vertices in strip data array
  int doublesided; ///< whether or not we need double-sided lighting
};

/// draw a wire mesh consisting of vertices, facets, colors, normals etc.
struct DispCmdWireMesh
{
  static void
  putdata (const float * vertices, const float * normals, const float * colors,
      int num_verts, const int * lines, int num_lines, VMDDisplayList *);

  inline void
  getpointers (float *& cnv, ///< array of colors, normals, vertices
      int *& l ///< line vertex index array
      ) const
  {
    char *rawptr = (char *) this;
    cnv = (float *) (rawptr + sizeof(DispCmdWireMesh));
    l = (int *) (rawptr + sizeof(DispCmdWireMesh)
        + sizeof(float) * numverts * 10);
  }

  int numverts; ///< number of vertices in mesh
  int numlines; ///< total number of lines
};

#define CYLINDER_TRAILINGCAP 1
#define CYLINDER_LEADINGCAP  2

/// draw a cylinder between two endpoints
struct DispCmdCylinder
{
  DispCmdCylinder (void);
  float rot[2]; ///< cache the angle computation
  int lastres;
  void
  putdata (const float *, const float *, float, int, int filled,
      VMDDisplayList *);
};

/// plot a cone at the given position
struct DispCmdCone
{
  void
  putdata (float*, float *, float, float, int, VMDDisplayList *);
  float pos1[3], pos2[3];
  float radius, radius2;
  int res;
};

/// set the current drawing color to the specified index
struct DispCmdColorIndex
{
  void
  putdata (int, VMDDisplayList *);
  int color;
};

/// display 3-D text at the given text coordinates
struct DispCmdText
{
  void
  putdata (const float *, const char *, VMDDisplayList *);
};

/// add a comment to the display list, to make token output etc. meaningful.  
struct DispCmdComment
{
  void
  putdata (const char *, VMDDisplayList *);
};

/// change the current text size
struct DispCmdTextSize
{
  void
  putdata (float, VMDDisplayList *);
  float size;
};

/// add an offset to rendered text
struct DispCmdTextOffset
{
  void
  putdata (float x, float y, VMDDisplayList *);
  float x, y;
};

/// apply a 3-D texture to a slice through the current volume texture
struct DispCmdVolSlice
{
  void
  putdata (int mode, const float *norm, const float *verts,
      const float *texcoords, VMDDisplayList *);
  /// volume slice plane normal (for shading)
  int texmode; ///< 3-D texture filtering mode (nearest/linear)
  float normal[3];
  /// vertexes and texture coordinates for volume slice plane
  float v[12];
  float t[12];
};

/// tell OpenGL to cache a 3D texture with the given unique ID.  Memory for
/// the texture is retained by the caller, not copied into the display list.
struct DispCmdVolumeTexture
{
  void
  putdata (unsigned long texID, const int size[3], unsigned char *texptr,
      const float vp0[3], const float vp1[3], const float vp2[3],
      const float vp3[3], VMDDisplayList *);
  unsigned char * texmap; ///< 3-D texture map
  unsigned long ID; ///< serial number for this 3-D texture
  unsigned xsize;
  unsigned ysize;
  unsigned zsize;
  float v0[3]; ///< origin for texgen plane equation
  float v1[3]; ///< X axis for texgen plane equation
  float v2[3]; ///< Y axis for texgen plane equation
  float v3[3]; ///< Z axis for texgen plane equation
};

/// set the current sphere resolution
struct DispCmdSphereRes
{
  void
  putdata (int, VMDDisplayList *);
  int res;
};

/// set the current sphere type
struct DispCmdSphereType
{
  void
  putdata (int, VMDDisplayList *);
  int type;
};

/// set the current line type
struct DispCmdLineType
{
  void
  putdata (int, VMDDisplayList *);
  int type;
};

/// set the current line width
struct DispCmdLineWidth
{
  void
  putdata (int, VMDDisplayList *);
  int width;
};

/// indicate a point which may be picked, and the associated 'tag'
struct DispCmdPickPoint
{
  void
  putdata (float *, int, VMDDisplayList *);
  float postag[3];
  int tag;
};

/// indicate a point which may be picked, and the 'tag' which is associated
/// with this point.  This uses an indexed position instead of a specific one.
struct DispCmdPickPointIndex
{
  void
  putdata (int, int, VMDDisplayList *);
  int pos, tag;
};

/// Create an array of indexed points which may be picked and their associated
/// tags.  
struct DispCmdPickPointIndexArray
{
  /// Create the pick point list from an atom selection.
  /// In the case where all indices are enabled, the implementation 
  /// doesn't store the array, for increased memory efficiency.  In the case
  /// where only some indices are selected, an explicit list is constructed.
  void
  putdata (int num, int numsel, int *onoff, VMDDisplayList *);

  /// Create the pick point list from an existing array of pick point indices
  void
  putdata (int num, int *indices, VMDDisplayList *);

  inline void
  getpointers (int *& tags) const
  {
    if (allselected)
    {
      tags = NULL;
      return;
    }

    char *rawptr = (char *) this;
    tags = (int *) (rawptr + sizeof(DispCmdPickPointIndexArray));
  }

  int numpicks;
  int allselected;
};

#endif

