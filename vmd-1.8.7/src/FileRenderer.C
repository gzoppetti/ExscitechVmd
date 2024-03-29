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
 *	$RCSfile: FileRenderer.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.127 $	$Date: 2009/06/05 20:54:28 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * The FileRenderer class implements the data and functions needed to 
 * render a scene to a file in some format (postscript, raster3d, etc.)
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"
#include "DispCmds.h"
#include "FileRenderer.h"
#include "VMDDisplayList.h"
#include "Inform.h"
#include "Scene.h"
#include "Hershey.h"

// constructor
FileRenderer::FileRenderer(const char *public_name, 
                           const char *default_file_name,
			   const char *default_command_line) : 
  DisplayDevice(public_name), transMat(10)
{
  // save the various names
  publicName = stringdup(public_name);
  defaultFilename = stringdup(default_file_name);
  defaultCommandLine = stringdup(default_command_line);
  execCmd = stringdup(defaultCommandLine);
  has_aa = 0;
  aasamples = -1;
  has_imgsize = 0;
  imgwidth = imgheight = 0;
  aspectratio = 0.0f;
  curformat = -1;
  warningflags = FILERENDERER_NOWARNINGS;

  // init some state variables
  outfile = NULL;
  isOpened = FALSE;
  my_filename = NULL;

  // initialize sphere tesselation variables
  sph_nverts = 0;
  sph_verts = NULL;
}

// close (to be on the safe side) and delete
FileRenderer::~FileRenderer(void) {
  // free sphere tessellation data
  if (sph_verts && sph_nverts) 
    free(sph_verts);

  close_file();
  delete [] my_filename;
  delete [] publicName;
  delete [] defaultFilename;
  delete [] defaultCommandLine;
  delete [] execCmd;
}

int FileRenderer::do_define_light(int n, float *color, float *position) {
  for (int i=0; i<3; i++) {
    lightState[n].color[i] = color[i];
    lightState[n].pos[i] = position[i];
  }
  return TRUE;
}

int FileRenderer::do_activate_light(int n, int turnon) {
  lightState[n].on = turnon;
  return TRUE;
}

void FileRenderer::do_use_colors() {
  for (int i=0; i<MAXCOLORS; i++) {
    matData[i][0] = colorData[3*i  ];
    matData[i][1] = colorData[3*i+1];
    matData[i][2] = colorData[3*i+2];
  }
}

int FileRenderer::set_imagesize(int *w, int *h) {
  if (*w < 0 || *h < 0) return FALSE;
  if (*w == imgwidth && *h == imgheight) return TRUE;
  if (!aspectratio) {
    if (*w) imgwidth = *w; 
    if (*h) imgheight = *h;
  } else {
    if (*w) {
      imgwidth = *w;
      imgheight = (int)(*w / aspectratio);
    } else if (*h) {
      imgwidth = (int)(*h * aspectratio);
      imgheight = *h;
    } else {
      if (imgwidth || imgheight) {
        int wtmp = imgwidth, htmp = imgheight;
        set_imagesize(&wtmp, &htmp);
      }
    }
  }
  update_exec_cmd();
  *w = imgwidth;
  *h = imgheight;
  return TRUE;
}

float FileRenderer::set_aspectratio(float aspect) {
  if (aspect >= 0) aspectratio = aspect;
  int w=0, h=0;
  set_imagesize(&w, &h);  // update_exec_cmd() called from set_imagesize() 
  return aspectratio;
}

int FileRenderer::nearest_index(float r, float g, float b) const {
   const float *rcol = matData[BEGREGCLRS];  // get the solid colors
   float lsq = r - rcol[0]; lsq *= lsq;
   float tmp = g - rcol[1]; lsq += tmp * tmp;
         tmp = b - rcol[2]; lsq += tmp * tmp;
   float best = lsq;
   int bestidx = BEGREGCLRS;
   for (int n=BEGREGCLRS+1; n < (BEGREGCLRS + REGCLRS + MAPCLRS); n++) {
     rcol = matData[n];
     lsq = r - rcol[0]; lsq *= lsq;
     tmp = g - rcol[1]; lsq += tmp * tmp;
     tmp = b - rcol[2]; lsq += tmp * tmp;
     if (lsq < best) {
       best = lsq;
       bestidx = n;
     }
   }
   return bestidx;
}

void FileRenderer::set_background(const float * bg) {
  backColor[0] = bg[0];
  backColor[1] = bg[1];
  backColor[2] = bg[2];
}


// open file, closing the previous file if it was still open
int FileRenderer::open_file(const char *filename) {
  if (isOpened) {
    close_file();
  }
  if ( (outfile = fopen(filename, "w")) == NULL) {
    msgErr << "Could not open file " << filename
           << " in current directory for writing!" << sendmsg;
    return FALSE;
  }
  my_filename = stringdup(filename);
  isOpened = TRUE;
  reset_state();
  return TRUE;
}

void FileRenderer::reset_state(void) {
  // empty out the viewing stack
  while (transMat.num()) {
    transMat.pop();
  }
  // reset everything else
  colorIndex = -1;
  materialIndex = -1;
  dataBlock = NULL;
  lineWidth = 1;
  lineStyle = 1;
  pointSize = 1;
  sphereResolution = 4;
  sphereStyle = 1;
  materials_on = 0;
}

// close the file.  This could be called by open_file if the previous
// file wasn't closed, so don't put too much here
void FileRenderer::close_file(void) {
  if (outfile) {
    fclose(outfile);
    outfile = NULL;
  }
  delete [] my_filename;
  my_filename = NULL;
  isOpened = FALSE;
}


int FileRenderer::prepare3D(int) {
  // set the eye position, based on the value of whichEye, which was
  // obtained when we copied the current visible display device to the
  // file renderer.  
  int i;
  float lookat[3];
  for (i=0; i<3; i++) 
    lookat[i] = eyePos[i] + eyeDir[i];

  switch (whichEye) {
    case LEFTEYE:
      for (i=0; i<3; i++) 
        eyePos[i] -= eyeSepDir[i];
  
      for (i=0; i<3; i++) 
        eyeDir[i] = lookat[i] - eyePos[i];
      break; 

    case RIGHTEYE:
      for (i=0; i<3; i++) 
        eyePos[i] += eyeSepDir[i]; 

      for (i=0; i<3; i++) 
        eyeDir[i] = lookat[i] - eyePos[i];
      break;

    case NOSTEREO: 
      break;
  }

  if (isOpened) {
    write_header();
  }

  return TRUE;
}

/////////////////////////////////// render the display lists

void FileRenderer::render(const VMDDisplayList *cmdList) {
  if (!cmdList) return;
  int tok, i;
  char *cmdptr; 

  // scan through the list and do the action based on the token type
  // if the command relates to the viewing state, keep track of it
  // for those renderers that don't store state
  while (transMat.num()) {    // clear the stack
    transMat.pop();
  }
  Matrix4 m;
  transMat.push(m);           // push on the identity matrix
  super_multmatrix(cmdList->mat.mat);

  colorIndex = 0;
  materialIndex = 0;
  lineWidth = 1;
  lineStyle = 1;
  pointSize = 1;
  sphereResolution = 4;
  sphereStyle = 1;
 
  // set the material properties
  super_set_material(cmdList->materialtag);
  mat_ambient   = cmdList->ambient;
  mat_specular  = cmdList->specular;
  mat_diffuse   = cmdList->diffuse;
  mat_shininess = cmdList->shininess;
  mat_opacity   = cmdList->opacity;
  mat_outline   = cmdList->outline;
  mat_outlinewidth = cmdList->outlinewidth;

  for (i=0; i<VMD_MAX_CLIP_PLANE; i++) {
    clip_mode[i] = cmdList->clipplanes[i].mode;
    memcpy(&clip_center[i][0], &cmdList->clipplanes[i].center, 3*sizeof(float));
    memcpy(&clip_normal[i][0], &cmdList->clipplanes[i].normal, 3*sizeof(float));
    memcpy(&clip_color[i][0],  &cmdList->clipplanes[i].color,  3*sizeof(float));
  }
  start_clipgroup();

  // Compute periodic images
  ResizeArray<Matrix4> pbcImages;
  find_pbc_images(cmdList, pbcImages);
  int nimages = pbcImages.num();

for (int pbcimage = 0; pbcimage < nimages; pbcimage++) {
 transMat.dup();
 super_multmatrix(pbcImages[pbcimage].mat);

  VMDDisplayList::VMDLinkIter cmditer;
  cmdList->first(&cmditer);
  while ((tok = cmdList->next(&cmditer, cmdptr))  != DLASTCOMMAND) {
    switch (tok) {   // plot a point
    case DDATABLOCK:
#ifdef VMDCAVE
      dataBlock = (float *)cmdptr;
#else
      dataBlock = ((DispCmdDataBlock *)cmdptr)->data;
#endif
      break;                                                                  

    case DPOINT:
      point(((DispCmdPoint *)cmdptr)->pos);  
      break;

    case DPOINTARRAY:
      {
      int i, ind;
      DispCmdPointArray *pa = (DispCmdPointArray *)cmdptr;
      float *centers;
      float *colors;
      pa->getpointers(centers, colors);

      pointSize = (int) pa->size;     // set the point size
      ind = 0;
      for (i=0; i<pa->numpoints; i++) {
        super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2])); 
        point(&centers[ind]);  
        ind += 3;
      }
      pointSize = 1;            // reset the point size
      }
      break;

    case DLITPOINTARRAY:
      {
      int i, ind;
      DispCmdLitPointArray *pa = (DispCmdLitPointArray *)cmdptr;
      float *centers;
      float *normals;
      float *colors;
      pa->getpointers(centers, normals, colors);
      pointSize = (int) pa->size;     // set the point size
      ind = 0;
      for (i=0; i<pa->numpoints; i++) {
        super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2])); 
        point(&centers[ind]);  
        ind += 3;
      }
      pointSize = 1;            // reset the point size
      }
      break;

    case DSPHERE:    // draw a sphere
      sphere((float *)cmdptr);  
      break;

    case DSPHERE_I: {// plot a sphere using indices into dataBlock
      DispCmdSphereIndex *cmd = (DispCmdSphereIndex *)cmdptr;
      int n1 = cmd->pos; 
      float spheredata[4];
      spheredata[0] = dataBlock[n1++];
      spheredata[1] = dataBlock[n1++];
      spheredata[2] = dataBlock[n1];
      spheredata[3] = cmd->rad; 
      sphere(spheredata);
      break;
    }

    case DSPHEREARRAY:     
      {
      DispCmdSphereArray *sa = (DispCmdSphereArray *)cmdptr;
      int i, ind;
      float * centers;
      float * radii;
      float * colors;
      sa->getpointers(centers, radii, colors);

      set_sphere_res(sa->sphereres); // set the current sphere resolution

      ind = 0;
      for (i=0; i<sa->numspheres; i++) {
        float xyzr[4];
        xyzr[0]=centers[ind    ];
        xyzr[1]=centers[ind + 1];
        xyzr[2]=centers[ind + 2];
        xyzr[3]=radii[i];

        super_set_color(nearest_index(colors[ind], colors[ind+1], colors[ind+2])); 
        sphere(xyzr);
        ind += 3; // next sphere
      }
      }
      break;

    case DLINE:    // plot a line
      // don't draw degenerate lines of zero length
      if (memcmp(cmdptr, cmdptr+3*sizeof(float), 3*sizeof(float))) {
	line((float *)cmdptr, ((float *)cmdptr) + 3);
      }
      break;
   
    case DLINEARRAY: // array of lines
      {
      float *v = (float *) cmdptr;
      int nlines = (int)v[0];
      v++;
      for (int i=0; i<nlines; i++) {
        // don't draw degenerate lines of zero length
        if (memcmp(v,v+3,3*sizeof(float))) {
          line(v,v+3);
        }
        v += 6;
      }
      }
      break; 

    case DPOLYLINEARRAY: // array of lines
      {
      float *v = (float *) cmdptr;
      int nlines = (int)v[0];
      v++;
      for (int i=0; i<nlines-1; i++) {
        // don't draw degenerate lines of zero length
        if (memcmp(v,v+3,3*sizeof(float))) {
          line(v,v+3);
        }
        v += 3;
      }
      }
      break; 

    case DCYLINDER: // plot a cylinder
      if (memcmp(cmdptr, cmdptr+3*sizeof(float), 3*sizeof(float))) {
	cylinder((float *)cmdptr, ((float *)cmdptr) + 3, ((float *)cmdptr)[6],
		 ((int) ((float *) cmdptr)[8]));
      }
      break;

    case DCONE:      // plot a cone  
      {
      DispCmdCone *cmd = (DispCmdCone *)cmdptr;
      if (memcmp(cmd->pos1, cmd->pos2, 3*sizeof(float))) 
	cone(cmd->pos1, cmd->pos2, cmd->radius, cmd->radius2, cmd->res);
      }
      break;
   
    case DTRIANGLE:    // plot a triangle
      {
      DispCmdTriangle *cmd = (DispCmdTriangle *)cmdptr;
      triangle(cmd->pos1,cmd->pos2,cmd->pos3,
               cmd->norm1, cmd->norm2, cmd->norm3);
      }
      break;

    case DTRIMESH:     // draw a triangle mesh
      {
      DispCmdTriMesh *cmd = (DispCmdTriMesh *) cmdptr;
      float *cnv;
      int *f;
      cmd->getpointers(cnv, f);
      trimesh(cmd->numverts, cnv, cmd->numfacets, f); 
      }
      break;

    case DTRISTRIP:     // draw a triangle strip
      {
      DispCmdTriStrips *cmd = (DispCmdTriStrips *) cmdptr;
      float *cnv;
      int *f;
      int *vertsperstrip;
      cmd->getpointers(cnv, f, vertsperstrip);
      tristrip(cmd->numverts, cnv, cmd->numstrips, vertsperstrip, f); 
      }
      break;

    case DWIREMESH:     // draw a triangle mesh in wireframe
      {
      DispCmdWireMesh *cmd = (DispCmdWireMesh *) cmdptr;
      float *cnv;
      int *l;
      cmd->getpointers(cnv, l);
      wiremesh(cmd->numverts, cnv, cmd->numlines, l); 
      }
      break;

    case DSQUARE:      // plot a square (norm, 4 verticies
      {
      DispCmdSquare *cmd = (DispCmdSquare *)cmdptr;
      square(cmd->norml, cmd->pos1, cmd->pos2, cmd->pos3, cmd->pos4);
      }
      break;


    ///////////// keep track of state information as well
    case DLINEWIDTH:   //  set the line width (and in superclass)
      lineWidth = ((DispCmdLineWidth *)cmdptr)->width;
      set_line_width(lineWidth);
      break;

    case DLINESTYLE:   // set the line style (and in superclass)
      lineStyle = ((DispCmdLineType *)cmdptr)->type;
      set_line_style(lineStyle);
      break;

    case DSPHERERES:   // sphere resolution (and in superclass)
      sphereResolution = ((DispCmdSphereRes *)cmdptr)->res;
      set_sphere_res(sphereResolution);
      break;

    case DSPHERETYPE:   // sphere resolution (and in superclass)
      sphereStyle = ((DispCmdSphereType *)cmdptr)->type;
      set_sphere_style(sphereStyle);
      break;

    case DMATERIALON:
      super_materials(1); 
      break;

    case DMATERIALOFF:
      super_materials(0); 
      break;

    case DCOLORINDEX:  // change the color
      super_set_color(((DispCmdColorIndex *)cmdptr)->color);
      break; 

    case DTEXTSIZE:
      // not implemented yet
      warningflags |= FILERENDERER_NOTEXT;
      break;

    case DTEXT: 
      {
      float *pos = (float *)cmdptr;
      super_text_position(pos[0], pos[1], pos[2]);
      text((char *) (pos+3));
      }
      break;

    case DCOMMENT:
      comment((char *)cmdptr);
      break;

    // pick selections (only one implemented)
    case DPICKPOINT:
      pick_point(((DispCmdPickPoint *)cmdptr)->postag,
                 ((DispCmdPickPoint *)cmdptr)->tag); 
      break;

    case DPICKPOINT_I:
      pick_point(dataBlock + ((DispCmdPickPointIndex *)cmdptr)->pos,
                 ((DispCmdPickPointIndex *)cmdptr)->tag); 
      break;

    case DPICKPOINT_IARRAY:
      { 
        int i;
        DispCmdPickPointIndexArray *cmd =  ((DispCmdPickPointIndexArray *)cmdptr);
        if (cmd->allselected) {
          for (i=0; i<cmd->numpicks; i++) {
            pick_point(dataBlock + i*3, i);
          }
        } else {
          int *indices;
          cmd->getpointers(indices);
          for (i=0; i<cmd->numpicks; i++) {
            pick_point(dataBlock + indices[i]*3, indices[i]); 
          }
        }
      }
      break;

    // generate warnings if any geometry token is unimplemented the renderer
#if 0
    case DSTRIPETEXON:
    case DSTRIPETEXOFF:
#endif
    case DVOLUMETEXTURE:
      {
      DispCmdVolumeTexture *cmd = (DispCmdVolumeTexture *)cmdptr;
      float xplaneeq[4];
      float yplaneeq[4];
      float zplaneeq[4];
      int i;

      // automatically generate texture coordinates by translating from
      // model coordinate space to volume coordinates.
      for (i=0; i<3; i++) {
        xplaneeq[i] = cmd->v1[i];
        yplaneeq[i] = cmd->v2[i];
        zplaneeq[i] = cmd->v3[i];
      }
      xplaneeq[3] = cmd->v0[0];
      yplaneeq[3] = cmd->v0[1];
      zplaneeq[3] = cmd->v0[2];

      // define a volumetric texture map
      define_volume_texture(cmd->ID, cmd->xsize, cmd->ysize, cmd->zsize,
                            xplaneeq, yplaneeq, zplaneeq,
                            cmd->texmap);
      volume_texture_on(1);
      }
      break;

    case DVOLSLICE:
      {
      // Since OpenGL is using texture-replace here, we emulate that
      // by disabling lighting altogether
      super_materials(0); 
      DispCmdVolSlice *cmd = (DispCmdVolSlice *)cmdptr;
      volume_texture_on(1);
      square(cmd->normal, cmd->v, cmd->v + 3, cmd->v + 6, cmd->v + 9); 
      volume_texture_off();
      super_materials(1); 
      }
      break;

    case DVOLTEXON:
      volume_texture_on(0);
      break;

    case DVOLTEXOFF:
      volume_texture_off();
      break;

#if 0
    // generate warnings if any geometry token is unimplemented the renderer
    default:
      warningflags |= FILERENDERER_NOMISCFEATURE;
      break;
#endif
    } // end of switch statement
  } // while (tok != DLASTCOMMAND)
 transMat.pop();
} // end of loop over periodic images
  end_clipgroup();
}

////////////////////////////////////////////////////////////////////


// change the active color
void FileRenderer::super_set_color(int color_index) {
  if (colorIndex != color_index) {
    colorIndex = color_index;
    set_color(color_index);
  }
}

// change the active material
void FileRenderer::super_set_material(int material_index) {
  if (materialIndex != material_index) {
    materialIndex = material_index;
    set_material(material_index);
  }
}

// turn materials on or off
void FileRenderer::super_materials(int on_or_off) {
  if (on_or_off) {
    materials_on = 1;
    activate_materials();
  } else {
    materials_on = 0;
    deactivate_materials();
  }
}


//////////////// change the viewing matrix array state ////////////////

void FileRenderer::super_load(float *cmdptr) {
  Matrix4 tmp(cmdptr);
  (transMat.top()).loadmatrix(tmp);
  load(tmp);
}
void FileRenderer::super_multmatrix(const float *cmdptr) {
  Matrix4 tmp(cmdptr);
  (transMat.top()).multmatrix(tmp);
  multmatrix(tmp);
}

void FileRenderer::super_translate(float *cmdptr) {
  (transMat.top()).translate( cmdptr[0], cmdptr[1], cmdptr[2]);
  translate( cmdptr[0], cmdptr[1], cmdptr[2]);
}

void FileRenderer::super_rot(float * cmdptr) {
  (transMat.top()).rot( cmdptr[0], 'x' + (int) (cmdptr[1]) );
  rot( cmdptr[0], 'x' + (int) (cmdptr[1]) );
}

void FileRenderer::super_scale(float *cmdptr) {
  (transMat.top()).scale( cmdptr[0], cmdptr[1], cmdptr[2] );
  scale( cmdptr[0], cmdptr[1], cmdptr[2] );
}

void FileRenderer::super_scale(float s) {
  transMat.top().scale(s,s,s);
  scale(s,s,s);
}

// scale the radius a by the global scaling factor, return as b.
float FileRenderer::scale_radius(float r) {
  // of course, VMD does not have a direction-independent scaling
  // factor, so I'll fake it using an average of the scaling
  // factors in each direction.
  
  float scaleFactor;
  
  scaleFactor = (sqrtf( 
  (((transMat.top()).mat)[0])*(((transMat.top()).mat)[0]) +
  (((transMat.top()).mat)[4])*(((transMat.top()).mat)[4]) +
  (((transMat.top()).mat)[8])*(((transMat.top()).mat)[8]) ) +
  sqrtf( (((transMat.top()).mat)[1])*(((transMat.top()).mat)[1]) +
  (((transMat.top()).mat)[5])*(((transMat.top()).mat)[5]) +
  (((transMat.top()).mat)[9])*(((transMat.top()).mat)[9]) ) +
  sqrtf( (((transMat.top()).mat)[2])*(((transMat.top()).mat)[2]) +
  (((transMat.top()).mat)[6])*(((transMat.top()).mat)[6]) +
  (((transMat.top()).mat)[10])*(((transMat.top()).mat)[10]) ) ) / 3.0f;
  
  if(r < 0.0) {
    msgErr << "FileRenderer: Error, Negative radius" << sendmsg;
    r = -r;
  } 
  
  r = r * scaleFactor;
  
  return r;
}


///////////////////////////// text
void FileRenderer::super_text_position(float x, float y, float z) {
  textpos[0] = x;
  textpos[1] = y;
  textpos[2] = z;
  text_position(x, y, z);
}

////// set the exec command 
void FileRenderer::set_exec_string(const char *extstr) {
  delete [] execCmd;
  execCmd = stringdup(extstr);
}


// default triangulated implementation of two-radius cones
void FileRenderer::cone(float *base, float *apex, float radius, float radius2, int numsides) {
  int h;
  float theta, incTheta, cosTheta, sinTheta;
  float axis[3], temp[3], perp[3], perp2[3];
  float vert0[3], vert1[3], vert2[3], edge0[3], edge1[3], face0[3], face1[3], norm0[3], norm1[3];

  axis[0] = base[0] - apex[0];
  axis[1] = base[1] - apex[1];
  axis[2] = base[2] - apex[2];
  vec_normalize(axis);

  // Find an arbitrary vector that is not the axis and has non-zero length
  temp[0] = axis[0] - 1.0f;
  temp[1] = 1.0f;
  temp[2] = 1.0f;

  // use the cross product to find orthoganal vectors
  cross_prod(perp, axis, temp);
  vec_normalize(perp);
  cross_prod(perp2, axis, perp); // shouldn't need normalization

  // Draw the triangles
  incTheta = (float) VMD_TWOPI / numsides;
  theta = 0.0;

  if (radius2>0) {
    float negaxis[3], offsetL[3], offsetT[3], vert3[3];
    int filled=1;
    vec_negate(negaxis, axis);
    memset(vert0, 0, sizeof(vert0));
    memset(vert1, 0, sizeof(vert1));
    memset(norm0, 0, sizeof(norm0));

    for (h=0; h <= numsides+3; h++) {
      cosTheta = (float) cos(theta);
      sinTheta = (float) sin(theta);
      offsetL[0] = radius2 * (cosTheta*perp[0] + sinTheta*perp2[0]);
      offsetL[1] = radius2 * (cosTheta*perp[1] + sinTheta*perp2[1]);
      offsetL[2] = radius2 * (cosTheta*perp[2] + sinTheta*perp2[2]);
      offsetT[0] = radius  * (cosTheta*perp[0] + sinTheta*perp2[0]);
      offsetT[1] = radius  * (cosTheta*perp[1] + sinTheta*perp2[1]);
      offsetT[2] = radius  * (cosTheta*perp[2] + sinTheta*perp2[2]);

      // copy old vertices
      vec_copy(vert2, vert0); 
      vec_copy(vert3, vert1); 
      vec_copy(norm1, norm0); 

      // calculate new vertices
      vec_add(vert0, base, offsetT);
      vec_add(vert1, apex, offsetL);

      // Use the new vertex to find new edges
      edge0[0] = vert0[0] - vert1[0];
      edge0[1] = vert0[1] - vert1[1];
      edge0[2] = vert0[2] - vert1[2];
      edge1[0] = vert0[0] - vert2[0];
      edge1[1] = vert0[1] - vert2[1];
      edge1[2] = vert0[2] - vert2[2];

      // Use the new edge to find a new facet normal
      cross_prod(norm0, edge1, edge0);
      vec_normalize(norm0);

      if (h > 2) {
	// Use the new normal to draw the previous side
	triangle(vert0, vert1, vert3, norm0, norm0, norm1);
	triangle(vert3, vert2, vert0, norm1, norm1, norm0);

	// Draw cylinder caps
	if (filled & CYLINDER_LEADINGCAP) {
	  triangle(vert1, vert3, apex, axis, axis, axis);
	}
	if (filled & CYLINDER_TRAILINGCAP) {
	  triangle(vert0, vert2, base, negaxis, negaxis, negaxis);
	}
      }

      theta += incTheta;
    }
  } else {
    for (h=0; h < numsides+3; h++) {
      cosTheta = (float) cos(theta);
      sinTheta = (float) sin(theta);

      // Find a new vertex
      vert0[0] = base[0] + radius * ( cosTheta*perp[0] + sinTheta*perp2[0] );
      vert0[1] = base[1] + radius * ( cosTheta*perp[1] + sinTheta*perp2[1] );
      vert0[2] = base[2] + radius * ( cosTheta*perp[2] + sinTheta*perp2[2] );

      // Use the new vertex to find a new edge
      edge0[0] = vert0[0] - apex[0];
      edge0[1] = vert0[1] - apex[1];
      edge0[2] = vert0[2] - apex[2];

      if (h > 0) {
	// Use the new edge to find a new face
	cross_prod(face0, edge1, edge0);
	vec_normalize(face0);

	if (h > 1) {
	  // Use the new face to find the normal of the previous triangle
	  norm0[0] = (face1[0] + face0[0]) / 2.0f;
	  norm0[1] = (face1[1] + face0[1]) / 2.0f;
	  norm0[2] = (face1[2] + face0[2]) / 2.0f;
	  vec_normalize(norm0);

	  if (h > 2) {
	    // Use the new normal to draw the previous side and base of the cone
	    triangle(vert2, vert1, apex, norm1, norm0, face1);
	    triangle(vert2, vert1, base, axis, axis, axis);
	  }

	}
	// Copy the old values
	memcpy(norm1, norm0, 3*sizeof(float));
	memcpy(vert2, vert1, 3*sizeof(float));
	memcpy(face1, face0, 3*sizeof(float));
      }
      memcpy(vert1, vert0, 3*sizeof(float));
      memcpy(edge1, edge0, 3*sizeof(float));
  
      theta += incTheta;
    }
  }
}


// default trianglulated cylinder implementation, with optional end caps
void FileRenderer::cylinder(float *base, float *apex, float radius, int filled) {
  const int numsides = 20;
  int h;
  float theta, incTheta, cosTheta, sinTheta;
  float axis[3], negaxis[3], temp[3], perp[3], perp2[3];
  float vert0[3], vert1[3], vert2[3], vert3[3];
  float offset[3], norm0[3], norm1[3];

  axis[0] = base[0] - apex[0];
  axis[1] = base[1] - apex[1];
  axis[2] = base[2] - apex[2];
  vec_normalize(axis);
  vec_negate(negaxis, axis);

  // Find an arbitrary vector that is not the axis and has non-zero length
  temp[0] = axis[0] - 1.0f;
  temp[1] = 1.0f;
  temp[2] = 1.0f;

  // use the cross product to find orthoganal vectors
  cross_prod(perp, axis, temp);
  vec_normalize(perp);
  cross_prod(perp2, axis, perp); // shouldn't need normalization

  // Draw the triangles
  incTheta = (float) VMD_TWOPI / numsides;
  theta = 0.0;

  memset(vert0, 0, sizeof(vert0));
  memset(vert1, 0, sizeof(vert1));
  memset(norm0, 0, sizeof(norm0));

  for (h=0; h <= numsides; h++) {
    cosTheta = (float) cos(theta);
    sinTheta = (float) sin(theta);
    offset[0] = radius * (cosTheta*perp[0] + sinTheta*perp2[0]);
    offset[1] = radius * (cosTheta*perp[1] + sinTheta*perp2[1]);
    offset[2] = radius * (cosTheta*perp[2] + sinTheta*perp2[2]);

    // copy old vertices
    vec_copy(vert2, vert0); 
    vec_copy(vert3, vert1); 
    vec_copy(norm1, norm0); 

    // calculate new vertices
    vec_add(vert0, base, offset);
    vec_add(vert1, apex, offset);

    // new normal
    vec_copy(norm0, offset);
    vec_normalize(norm0);

    if (h > 0) {
      // Use the new normal to draw the previous side
      triangle(vert0, vert1, vert3, norm0, norm0, norm1);
      triangle(vert3, vert2, vert0, norm1, norm1, norm0);

      // Draw cylinder caps
      if (filled & CYLINDER_LEADINGCAP) {
        triangle(vert1, vert3, apex, axis, axis, axis);
      }
      if (filled & CYLINDER_TRAILINGCAP) {
        triangle(vert0, vert2, base, negaxis, negaxis, negaxis);
      }
    }

    theta += incTheta;
  }
}


// default cylinder-based implementation of lines used
// for ray tracing packages that can't draw real lines
void FileRenderer::line(float * a, float * b) {
  // draw a line (cylinder) from a to b
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3]; 

  if (lineStyle == ::SOLIDLINE) {
    cylinder(a, b, lineWidth * 0.002f, 0);
  } else if (lineStyle == ::DASHEDLINE) {
    vec_sub(dirvec, b, a);        // vector from a to b
    vec_copy(unitdirvec, dirvec);
    vec_normalize(unitdirvec);    // unit vector from a to b
    test = 1;
    i = 0;
    while (test == 1) {
      for (j=0; j<3; j++) {
        from[j] = (float) (a[j] + (2*i    )* 0.05 * unitdirvec[j]);
          to[j] = (float) (a[j] + (2*i + 1)* 0.05 * unitdirvec[j]);
      }
      if (fabsf(a[0] - to[0]) >= fabsf(dirvec[0])) {
        vec_copy(to, b);
        test = 0;
      }
      cylinder(from, to, lineWidth * 0.002f, 0);
      i++;
    }
  } 
}


// default triangulated sphere implementation
void FileRenderer::sphere(float * xyzr) {
  float c[3], r;
  int pi, ni;
  int i;
  int sph_iter = -1;
  int sph_desired_iter = 0;

  // copy params
  vec_copy(c, xyzr);
  r = xyzr[3];

  // the sphere resolution has changed. if sphereRes is less than 32, we
  // will use a lookup table to achieve equal or better resolution than
  // OpenGL. otherwise we use the following equation:
  //    iterations = .9 *
  //    (sphereRes)^(1/2)
  // This is used as a lookup table to determine the proper
  // number of iterations used in the sphere approximation
  // algorithm.
  const int sph_iter_table[] = {
      0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4 };

  if (sphereResolution < 0) 
    return;
  else if (sphereResolution < 32) 
    sph_desired_iter = sph_iter_table[sphereResolution];
  else 
    sph_desired_iter = (int) (0.8f * sqrtf((float) sphereResolution));
 
  // first we need to determine if a recalculation of the cached
  // unit sphere is necessary. this is necessary if the number
  // of desired iterations has changed.
  if (!sph_verts || !sph_nverts || sph_iter != sph_desired_iter) {
    float a[3], b[3], c[3];
    float *newverts;
    float *oldverts;
    int nverts, ntris;
    int level;

    // remove old cached copy
    if (sph_verts && sph_nverts) free(sph_verts);

    newverts = (float *) malloc(sizeof(float) * 36);
    nverts = 12;
    ntris = 4;

    // start with half of a unit octahedron (front, convex half)

    // top left triangle
    newverts[0] = -1;    newverts[1] = 0;     newverts[2] = 0;
    newverts[3] = 0;     newverts[4] = 1;     newverts[5] = 0;
    newverts[6] = 0;     newverts[7] = 0;     newverts[8] = 1;

    // top right triangle
    newverts[9] = 0;     newverts[10] = 0;    newverts[11] = 1;
    newverts[12] = 0;    newverts[13] = 1;    newverts[14] = 0;
    newverts[15] = 1;    newverts[16] = 0;    newverts[17] = 0;

    // bottom right triangle
    newverts[18] = 0;    newverts[19] = 0;    newverts[20] = 1;
    newverts[21] = 1;    newverts[22] = 0;    newverts[23] = 0;
    newverts[24] = 0;    newverts[25] = -1;   newverts[26] = 0;

    // bottom left triangle
    newverts[27] = 0;    newverts[28] = 0;    newverts[29] = 1;
    newverts[30] = 0;    newverts[31] = -1;   newverts[32] = 0;
    newverts[33] = -1;   newverts[34] = 0;    newverts[35] = 0;

    for (level = 1; level < sph_desired_iter; level++) {
      oldverts = newverts;

      // allocate memory for the next iteration: we will need
      // four times the current number of vertices
      newverts = (float *) malloc(sizeof(float) * 12 * nverts);
      if (!newverts) {
        // memory error
        sph_iter = -1;
        sph_nverts = 0;
        sph_verts = NULL;
        free(oldverts);
        msgErr << "FileRenderer::sphere(): Out of memory. Some "
               << "objects were not drawn." << sendmsg;
        return;
      }

      pi = 0;
      ni = 0;
      for (i = 0; i < ntris; i++) {
        // compute intermediate vertices
        a[0] = (oldverts[pi    ] + oldverts[pi + 6]) / 2;
        a[1] = (oldverts[pi + 1] + oldverts[pi + 7]) / 2;
        a[2] = (oldverts[pi + 2] + oldverts[pi + 8]) / 2;
        vec_normalize(a);
        b[0] = (oldverts[pi    ] + oldverts[pi + 3]) / 2;
        b[1] = (oldverts[pi + 1] + oldverts[pi + 4]) / 2;
        b[2] = (oldverts[pi + 2] + oldverts[pi + 5]) / 2;
        vec_normalize(b);
        c[0] = (oldverts[pi + 3] + oldverts[pi + 6]) / 2;
        c[1] = (oldverts[pi + 4] + oldverts[pi + 7]) / 2;
        c[2] = (oldverts[pi + 5] + oldverts[pi + 8]) / 2;
        vec_normalize(c);

        // build triangles
        memcpy(&newverts[ni     ], &oldverts[pi], sizeof(float) * 3);
        memcpy(&newverts[ni + 3 ], b, sizeof(float) * 3);
        memcpy(&newverts[ni + 6 ], a, sizeof(float) * 3);

        memcpy(&newverts[ni + 9 ], b, sizeof(float) * 3);
        memcpy(&newverts[ni + 12], &oldverts[pi + 3], sizeof(float) * 3);
        memcpy(&newverts[ni + 15], c, sizeof(float) * 3);

        memcpy(&newverts[ni + 18], a, sizeof(float) * 3);
        memcpy(&newverts[ni + 21], b, sizeof(float) * 3);
        memcpy(&newverts[ni + 24], c, sizeof(float) * 3);

        memcpy(&newverts[ni + 27], a, sizeof(float) * 3);
        memcpy(&newverts[ni + 30], c, sizeof(float) * 3);
        memcpy(&newverts[ni + 33], &oldverts[pi + 6], sizeof(float) * 3);

        pi += 9;
        ni += 36;
      }

      free(oldverts);
      nverts *= 4;
      ntris *= 4;
    }

    sph_iter = sph_desired_iter;
    sph_nverts = nverts;
    sph_verts = newverts;
  }

  // now we're guaranteed to have a valid cached unit sphere, so
  // all we need to do is translate each coordinate based on the
  // desired position and radius, and add the triangles
  pi = 0;
  for (i = 0; i < sph_nverts / 3; i++) {
    float v0[3], v1[3], v2[3];
    float n0[3], n1[3], n2[3];

    // calculate upper hemisphere translation and scaling
    v0[0] = r * sph_verts[pi    ] + c[0];
    v0[1] = r * sph_verts[pi + 1] + c[1];
    v0[2] = r * sph_verts[pi + 2] + c[2];
    v1[0] = r * sph_verts[pi + 3] + c[0];
    v1[1] = r * sph_verts[pi + 4] + c[1];
    v1[2] = r * sph_verts[pi + 5] + c[2];
    v2[0] = r * sph_verts[pi + 6] + c[0];
    v2[1] = r * sph_verts[pi + 7] + c[1];
    v2[2] = r * sph_verts[pi + 8] + c[2];

    // calculate upper hemisphere normals
    vec_copy(n0, &sph_verts[pi    ]);
    vec_copy(n1, &sph_verts[pi + 3]);
    vec_copy(n2, &sph_verts[pi + 6]);

    // draw upper hemisphere
    triangle(v0, v1, v2, n0, n1, n2);

    // calculate lower hemisphere translation and scaling
    v0[2] = (-r * sph_verts[pi + 2]) + c[2];
    v1[2] = (-r * sph_verts[pi + 5]) + c[2];
    v2[2] = (-r * sph_verts[pi + 8]) + c[2];

    // calculate lower hemisphere normals
    n0[2] = -n0[2];
    n1[2] = -n1[2];
    n2[2] = -n2[2];

    // draw lower hemisphere
    triangle(v0, v2, v1, n0, n2, n1);

    pi += 9;
  }
}


// start rendering geometry for which user-defined
// clipping planes have been applied.
void FileRenderer::start_clipgroup() {
  int i;

  for (i=0; i<VMD_MAX_CLIP_PLANE; i++) {
    if (clip_mode[i]) {
      warningflags |= FILERENDERER_NOCLIP;
      break;
    }
  }
}


void FileRenderer::text(const char *str) {
#if 0
  hersheyhandle hh;
  float lm, rm, x, y, ox, oy;
  int draw, odraw;

  Matrix4 m;
  transMat.push(m);           // push on the identity matrix

  while (*str != '\0') {
    hersheyDrawInitLetter(&hh, *str, &lm, &rm);
    (transMat.top()).translate(-lm, 0, 0);
    ox=0;
    oy=0;
    odraw=0;
    while (!hersheyDrawNextLine(&hh, &draw, &x, &y)) {
      if (draw && odraw) {
        float a[3], b[3];
        a[0] = ox; 
        a[1] = oy;
        a[2] = 0;
        b[0] = x;
        b[1] = y;
        b[2] = 0;

  //      printf("line: %g %g -> %g %g\n", ox, oy, x, y);
        line(a, b);        
      }

      ox=x;
      oy=y;
      odraw=draw;
    }
    (transMat.top()).translate(rm, 0, 0);

    str++;
  }

  transMat.pop();
#endif
}


