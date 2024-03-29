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
 *	$RCSfile: DisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.123 $	$Date: 2009/05/17 06:37:37 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * DisplayDevice - abstract base class for all particular objects which
 *	can process a list of drawing commands and render the drawing
 *	to some device (screen, file, preprocessing script, etc.)
 *
 ***************************************************************************/

#include <math.h>
#include "DisplayDevice.h"
#include "Inform.h"
#include "DispCmds.h"
#include "utilities.h"
#include "Mouse.h"     // for WAIT_CURSOR
#include "VMDDisplayList.h"
#include "Scene.h"

// static data for this object (yuck)
static const char *cacheNameStr[1] =
  { "Off" };
static const char *renderNameStr[1] =
  { "Normal" };
static const char *stereoNameStr[1] =
  { "Stereo Off" };

const char *DisplayDevice::projNames[NUM_PROJECTIONS] =
  { "Perspective", "Orthographic" };

const char *DisplayDevice::cueModeNames[NUM_CUE_MODES] =
  { "Linear", "Exp", "Exp2" };

/////////////////////////  constructor and destructor  
DisplayDevice::DisplayDevice (const char *nm) :
    transMat (16)
{

  name = stringdup (nm); // save the string name of this display device
  num_display_processes = 1; // set number of rendering processes etc
  renderer_process = 1; // we're a rendering process until told otherwise
  _needRedraw = 0; // Start life not needing to be redrawn.

  // set default background drawing mode
  backgroundmode = 0;

  // set drawing characteristics default values
  lineStyle = ::SOLIDLINE;
  lineWidth = 1;
  sphereRes = 3;
  cylinderRes = 6;
  sphereMode = ::SOLIDSPHERE;

  // set scalar values
  aaAvailable = cueingAvailable = FALSE;
  aaPrevious = aaEnabled = cueingEnabled = FALSE;
  xOrig = yOrig = xSize = ySize = 0;
  screenX = screenY = 0;

  // set viewing geometry ... looking from z-axis in negative direction,
  // with 90-degree field of view and assuming the origin is in the
  // center of the viewer's 'screen'
  nearClip = 0.5f;
  farClip = 10.0f;
  eyePos[0] = eyePos[1] = 0.0f;
  eyePos[2] = 2.0f;
  set_screen_pos (2.0f * eyePos[2], 0.0f, 4.0f / 3.0f);

  // set initial depth cueing parameters 
  // (defaults are compatible with old revs of VMD)
  cueMode = CUE_EXP2;
  cueDensity = 0.4f;
  cueStart = 0.5f;
  cueEnd = 10.0f;

  // set initial shadow mode
  shadowEnabled = 0;

  // set initial ambient occlusion settings
  aoEnabled = 0;
  aoAmbient = 0.8;
  aoDirect = 0.3;

  // XXX stereo modes and rendering modes should be enumerated 
  // dynamically not hard-coded, to allow much greater flexibility

  // Setup stereo options ... while there is no stereo mode by default,
  // set up normal values for stereo data
  inStereo = 0;
  stereoModes = 1;
  stereoNames = stereoNameStr;

  // Setup caching mode options
  cacheMode = 0;
  cacheModes = 1;
  cacheNames = cacheNameStr;

  // Setup rendering mode options
  renderMode = 0;
  renderModes = 1;
  renderNames = renderNameStr;

  // default view/projection settings
  eyeSep = 0.065f; // default eye seperation
  eyeDist = eyePos[2]; // assumes viewer on pos z-axis

  float lookatorigin[3];
  vec_scale (&lookatorigin[0], -1, &eyePos[0]); // calc dir to origin

  // Exscitech: Normalize the eye direction!
  vec_normalize (lookatorigin);

  set_eye_dir (&lookatorigin[0]); // point camera at origin
  upDir[0] = upDir[2] = 0.0;
  upDir[1] = 1.0;
  calc_eyedir (); // determines eye separation direction
  my_projection = PERSPECTIVE;

  // load identity matrix onto top of transformation matrix stack
  Matrix4 temp_ident;
  transMat.push (temp_ident);

  mouseX = mouseY = 0;
}

// destructor
DisplayDevice::~DisplayDevice (void)
{
  set_stereo_mode (0); // return to non-stereo, if necessary
  delete[] name;
}

int
DisplayDevice::set_eye_defaults ()
{
  float defaultDir[3];
  float defaultPos[3] =
    { 0, 0, 2 }; // camera 2 units back from origin
  float defaultUp[3] =
    { 0, 1, 0 }; // Y is up

  vec_scale (&defaultDir[0], -1, &eyePos[0]); // calc dir to origin
  set_eye_dir (&defaultDir[0]); // point camera at origin

  set_eye_pos (&defaultPos[0]);
  set_eye_dir (&defaultDir[0]);
  set_eye_up (&defaultUp[0]);

  return TRUE;
}

/////////////////////////  protected nonvirtual routines  
// calculate the position of the near frustum plane, based on current values
// of Aspect, vSize, zDist, nearClip and eyePosition
// Assumes the viewer is looking toward the xy-plane

void
DisplayDevice::calc_frustum (void)
{
  float d;
  float halfvsize = 0.5f * vSize;
  float halfhsize = Aspect * halfvsize; // width = aspect * height

  // if the frustum parameters don't cause division by zero,
  // calculate the new view frustum

  float lengthSquared = eyePos[0] * eyePos[0] + eyePos[1] * eyePos[1] + eyePos[2] * eyePos[2];
  float length = sqrt(lengthSquared);

  //if (eyePos[2] - zDist != 0.0f)
  //{
    if (length - zDist != 0.0f)
    {
      // scaling ratio for the view frustum, essentially the amount of
      // perspective to apply.  Since we define the nearClip plane in
      // the user interface, we can control how strong the perspective
      // is by varying (eyePos[2] - zDist) or by scaling d by some other
      // user controllable factor.  In order to make this more transparent
      // to the user however, we'd need to automatically apply a scaling
      // operation on the molecular geometry so that it looks about the same
      // despite the perspective change.  We should also be able to calculate
      // the field of view angles (vertical, horizontal, and diagonal) based
      // on all of these variables.
      //d = nearClip / (eyePos[2] - zDist);
      d = nearClip / (length - zDist);

      cpRight = d * halfhsize; // right side is at half width
      cpLeft = -cpRight; // left side is at negative half width
      cpUp = d * halfvsize; // top side is at half height
      cpDown = -cpUp; // bottom is at negative half height
    }
  }

// calculate eyeSepDir, based on up vector and look vector
// eyeSepDir = 1/2 * eyeSep * (lookdir x updir) / mag(lookdir x updir)
  void
  DisplayDevice::calc_eyedir (void)
  {
    float *L = eyeDir;
    float *U = upDir;
    float m, A = 0.5f * eyeSep;
    eyeSepDir[0] = L[1] * U[2] - L[2] * U[1];
    eyeSepDir[1] = L[2] * U[0] - L[0] * U[2];
    eyeSepDir[2] = L[0] * U[1] - L[1] * U[0];
    m = sqrtf (
        eyeSepDir[0] * eyeSepDir[0] + eyeSepDir[1] * eyeSepDir[1]
            + eyeSepDir[2] * eyeSepDir[2]);
    if (m > 0.0)
      A /= m;
    else
      A = 0.0;
    eyeSepDir[0] *= A;
    eyeSepDir[1] *= A;
    eyeSepDir[2] *= A;
  }

/////////////////////////  public nonvirtual routines  

// Copy all relevant properties from one DisplayDevice to another
  DisplayDevice&
  DisplayDevice::operator= (DisplayDevice &display)
  {
    int i;

    xOrig = display.xOrig;
    yOrig = display.yOrig;
    xSize = display.xSize;
    ySize = display.ySize;

    // Do something about the stack.  For the moment, only copy the top
    // item on the stack.
    if (transMat.num () > 0)
    {
      transMat.pop ();
    }
    transMat.push ((display.transMat).top ());

    for (i = 0; i < 3; i++)
    {
      eyePos[i] = display.eyePos[i];
      eyeDir[i] = display.eyeDir[i];
      upDir[i] = display.upDir[i];
      eyeSepDir[i] = display.eyeSepDir[i];
    }

    whichEye = display.whichEye;
    nearClip = display.nearClip;
    farClip = display.farClip;
    vSize = display.vSize;
    zDist = display.zDist;
    Aspect = display.Aspect;
    cpUp = display.cpUp;
    cpDown = display.cpDown;
    cpLeft = display.cpLeft;
    cpRight = display.cpRight;
    inStereo = display.inStereo;
    eyeSep = display.eyeSep;
    eyeDist = display.eyeDist;
    lineStyle = display.lineStyle;
    lineWidth = display.lineWidth;
    my_projection = display.my_projection;
    cueingEnabled = display.cueingEnabled;
    cueMode = display.cueMode;
    cueDensity = display.cueDensity;
    cueStart = display.cueStart;
    cueEnd = display.cueEnd;
    shadowEnabled = display.shadowEnabled;
    aoEnabled = display.aoEnabled;
    aoAmbient = display.aoAmbient;
    aoDirect = display.aoDirect;
    return *this;
  }

/////////////////////////  public virtual routines

  void
  DisplayDevice::do_resize_window (int w, int h)
  {
    xSize = w;
    ySize = h;
    set_screen_pos ((float) xSize / (float) ySize);
  }

//
// event handling routines
//

// queue the standard events (need only be called once ... but this is
// not done automatically by the window because it may not be necessary or
// even wanted)
  void
  DisplayDevice::queue_events (void)
  {
    return;
  }

// read the next event ... returns an event type (one of the above ones),
// and a value.  Returns success, and sets arguments.
  int
  DisplayDevice::read_event (long &, long &)
  {
    return FALSE;
  }

//
// get the current state of the device's pointer (i.e. cursor if it has one)
//

// abs pos of cursor from lower-left corner
  int
  DisplayDevice::x (void)
  {
    return mouseX;
  }

// same, for y direction
  int
  DisplayDevice::y (void)
  {
    return mouseY;
  }

// the shift state (shift key, control key, and/or alt key)
  int
  DisplayDevice::shift_state (void)
  {
    return 0; // by default, nothing is down
  }

// set the Nth cursor shape as the current one.  If no arg given, the
// default shape (n=0) is used.
  void
  DisplayDevice::set_cursor (int)
  {
  }

// virtual functions to turn on/off depth cuing and antialiasing
  void
  DisplayDevice::aa_on (void)
  {
  }
  void
  DisplayDevice::aa_off (void)
  {
  }
  void
  DisplayDevice::cueing_on (void)
  {
  }
  void
  DisplayDevice::cueing_off (void)
  {
  }
  void
  DisplayDevice::culling_on (void)
  {
  }
  void
  DisplayDevice::culling_off (void)
  {
  }

// return absolute 2D screen coordinates, given 2D or 3D world coordinates.
  void
  DisplayDevice::abs_screen_loc_3D (float *wloc, long *sloc)
  {
    // just return world coords
    for (int i = 0; i < 2; i++)
      sloc[i] = (long) (wloc[i]);
  }

  void
  DisplayDevice::abs_screen_loc_2D (float *wloc, long *sloc)
  {
    // just return world coords
    for (int i = 0; i < 2; i++)
      sloc[i] = (long) (wloc[i]);
  }

// change to a different stereo mode (0 means 'off')
  void
  DisplayDevice::set_stereo_mode (int sm)
  {
    if (sm != 0)
    {
      msgErr << "DisplayDevice: Illegal stereo mode " << sm << " specified."
          << sendmsg;
    }
    else
    {
      inStereo = sm;
    }
  }

// change to a different rendering mode (0 means 'normal')
  void
  DisplayDevice::set_cache_mode (int sm)
  {
    if (sm != 0)
    {
      msgErr << "DisplayDevice: Illegal caching mode " << sm << " specified."
          << sendmsg;
    }
    else
    {
      cacheMode = sm;
    }
  }

// change to a different rendering mode (0 means 'normal')
  void
  DisplayDevice::set_render_mode (int sm)
  {
    if (sm != 0)
    {
      msgErr << "DisplayDevice: Illegal rendering mode " << sm << " specified."
          << sendmsg;
    }
    else
    {
      renderMode = sm;
    }
  }

// replace the current trans matrix with the given one
  void
  DisplayDevice::loadmatrix (const Matrix4 &m)
  {
    (transMat.top ()).loadmatrix (m);
  }

// multiply the current trans matrix with the given one
  void
  DisplayDevice::multmatrix (const Matrix4 &m)
  {
    (transMat.top ()).multmatrix (m);
  }

//
// virtual routines for preparing to draw, drawing, and finishing drawing
//

  int
  DisplayDevice::prepare3D (int)
  {
    return 1;
  } // ready to draw 3D
  void
  DisplayDevice::clear (void)
  {
  } // erase the device
  void
  DisplayDevice::left (void)
  { // ready to draw left eye
    whichEye = LEFTEYE;
  }
  void
  DisplayDevice::right (void)
  { // ready to draw right eye
    whichEye = RIGHTEYE;
  }
  void
  DisplayDevice::normal (void)
  { // ready to draw non-stereo
    whichEye = NOSTEREO;
  }
  void
  DisplayDevice::update (int)
  {
  } // finish up after drawing
  void
  DisplayDevice::reshape (void)
  {
  } // refresh device after change

// Grab the screen
  unsigned char *
  DisplayDevice::readpixels (int &x, int &y)
  {
    x = 0;
    y = 0;

    return NULL;
  }

  void
  DisplayDevice::find_pbc_images (const VMDDisplayList *cmdList,
      ResizeArray<Matrix4> &pbcImages)
  {
    if (cmdList->pbc == PBC_NONE)
    {
      pbcImages.append (Matrix4 ());
      return;
    }
    ResizeArray<int> pbcCells;
    find_pbc_cells (cmdList, pbcCells);
    for (int i = 0; i < pbcCells.num (); i += 3)
    {
      int nx = pbcCells[i];
      int ny = pbcCells[i + 1];
      int nz = pbcCells[i + 2];
      Matrix4 mat;
      for (int i1 = 1; i1 <= nx; i1++)
        mat.multmatrix (cmdList->transX);
      for (int i2 = -1; i2 >= nx; i2--)
        mat.multmatrix (cmdList->transXinv);
      for (int i3 = 1; i3 <= ny; i3++)
        mat.multmatrix (cmdList->transY);
      for (int i4 = -1; i4 >= ny; i4--)
        mat.multmatrix (cmdList->transYinv);
      for (int i5 = 1; i5 <= nz; i5++)
        mat.multmatrix (cmdList->transZ);
      for (int i6 = -1; i6 >= nz; i6--)
        mat.multmatrix (cmdList->transZinv);
      pbcImages.append (mat);
    }
  }

  void
  DisplayDevice::find_pbc_cells (const VMDDisplayList *cmdList,
      ResizeArray<int> &pbcCells)
  {
    int pbc = cmdList->pbc;
    if (pbc == PBC_NONE)
    {
      pbcCells.append (0);
      pbcCells.append (0);
      pbcCells.append (0);
    }
    else
    {
      int npbc = cmdList->npbc;
      int nx = pbc & PBC_X ? npbc : 0;
      int ny = pbc & PBC_Y ? npbc : 0;
      int nz = pbc & PBC_Z ? npbc : 0;
      int nox = pbc & PBC_OPX ? -npbc : 0;
      int noy = pbc & PBC_OPY ? -npbc : 0;
      int noz = pbc & PBC_OPZ ? -npbc : 0;
      int i, j, k;
      for (i = nox; i <= nx; i++)
      {
        for (j = noy; j <= ny; j++)
        {
          for (k = noz; k <= nz; k++)
          {
            if (!(pbc & PBC_NOSELF && !i && !j && !k))
            {
              pbcCells.append (i);
              pbcCells.append (j);
              pbcCells.append (k);
            }
          }
        }
      }
    }
  }

//
//*******************  the picking routine  *********************
//
// This scans the given command list until the end, finding which item is
// closest to the given pointer position.
//
// arguments are dimension of picking (2 or 3), position of pointer,
// draw command list, and returned distance from object to eye position.
// Returns ID code ('tag') for item closest to pointer, or (-1) if no pick.
// If an object is picked, the eye distance argument is set to the distance
// from the display's eye position to the object (after its position has been
// found from the transformation matrix).  If the value of the argument when
// 'pick' is called is <= 0, a pick will be generated if any item is near the
// pointer.  If the value of the argument is > 0, a pick will be generated
// only if an item is closer to the eye position than the value of the
// argument.
// For 2D picking, coordinates are relative position in window from
//	lower-left corner (both in range 0 ... 1)
// For 3D picking, coordinates are the world coords of the pointer.  They
//	are the coords of the pointer after its transformation matrix has been
//	applied, and these coordinates are compared to the coords of the objects
//	when their transformation matrices are applied.

// but first, a macro for returning the distance^2 from the eyepos to the
// given position
#define DTOEYE(x,y,z) ( (x-eyePos[0])*(x-eyePos[0]) + \
			(y-eyePos[1])*(y-eyePos[1]) + \
			(z-eyePos[2])*(z-eyePos[2]) )
#define DTOPOINT(x,y,z) ( (x-pos[0])*(x-pos[0]) + \
			(y-pos[1])*(y-pos[1]) + \
			(z-pos[2])*(z-pos[2]) )

  int
  DisplayDevice::pick (int dim, const float *pos, const VMDDisplayList *cmdList,
      float &eyedist, int *unitcell, float window_size)
  {
    char *cmdptr = NULL;
    int tok;
    float newEyeDist, currEyeDist = eyedist;
    int tag = (-1), inRegion, currTag;
    int minX = 0, minY = 0, maxX = 0, maxY = 0;
    float fminX = 0.0f, fminY = 0.0f, fminZ = 0.0f, fmaxX = 0.0f, fmaxY = 0.0f,
        fmaxZ = 0.0f;
    float pntpos[3];
    long cpos[2];

    if (!cmdList)
      return (-1);

    // initialize picking: find screen region to look for object
    if (dim == 2)
    {
      fminX = pos[0] - window_size;
      fmaxX = pos[0] + window_size;
      fminY = pos[1] - window_size;
      fmaxY = pos[1] + window_size;
      abs_screen_pos (fminX, fminY);
      abs_screen_pos (fmaxX, fmaxY);
      minX = (int) fminX;
      maxX = (int) fmaxX;
      minY = (int) fminY;
      maxY = (int) fmaxY;
    }
    else
    {
      fminX = pos[0] - window_size;
      fmaxX = pos[0] + window_size;
      fminY = pos[1] - window_size;
      fmaxY = pos[1] + window_size;
      fminZ = pos[2] - window_size;
      fmaxZ = pos[2] + window_size;
    }

    // make sure we do not disturb the regular transformation matrix
    transMat.dup ();
    (transMat.top ()).multmatrix (cmdList->mat);

    // Transform the current pick point for each periodic image
    ResizeArray<Matrix4> pbcImages;
    ResizeArray<int> pbcCells;
    find_pbc_images (cmdList, pbcImages);
    find_pbc_cells (cmdList, pbcCells);
    int pbcimg;
    for (pbcimg = 0; pbcimg < pbcImages.num (); pbcimg++)
    {
      transMat.dup ();
      (transMat.top ()).multmatrix (pbcImages[pbcimg]);

      // scan through the list, getting each command and executing it, until
      // the end of commands token is found
      VMDDisplayList::VMDLinkIter cmditer;
      cmdList->first (&cmditer);
      float *dataBlock = NULL;
      while ((tok = cmdList->next (&cmditer, cmdptr)) != DLASTCOMMAND)
      {
        switch (tok)
        {
          case DDATABLOCK:
#ifdef VMDCAVE
            dataBlock = (float *)cmdptr;
#else
            dataBlock = ((DispCmdDataBlock *) cmdptr)->data;
#endif
            break;

          case DPICKPOINT:
          case DPICKPOINT_I:
            // calculate the transformed position of the point
            if (tok == DPICKPOINT)
            {
              DispCmdPickPoint *cmd = (DispCmdPickPoint *) cmdptr;
              (transMat.top ()).multpoint3d (cmd->postag, pntpos);
              currTag = cmd->tag;
            }
            else
            {
              DispCmdPickPointIndex *cmd = (DispCmdPickPointIndex *) cmdptr;
              (transMat.top ()).multpoint3d (dataBlock + cmd->pos, pntpos);
              currTag = cmd->tag;
            }

            // check if in picking region ... different for 2D and 3D
            if (dim == 2)
            {
              // convert the 3D world coordinate to 2D absolute screen coord
              abs_screen_loc_3D (pntpos, cpos);

              // check to see if the position falls in our picking region
              inRegion = (cpos[0] >= minX && cpos[0] <= maxX && cpos[1] >= minY
                  && cpos[1] <= maxY);
            }
            else
            {
              // just check to see if the position is in a box centered on our
              // pointer.  The pointer position should already be transformed.
              inRegion = (pntpos[0] >= fminX && pntpos[0] <= fmaxX
                  && pntpos[1] >= fminY && pntpos[1] <= fmaxY
                  && pntpos[2] >= fminZ && pntpos[2] <= fmaxZ);
            }

            // has a hit occurred?
            if (inRegion)
            {
              // yes, see if it is closer to the eye position than earlier objects
              if (dim == 2)
                newEyeDist = DTOEYE(pntpos[0], pntpos[1], pntpos[2]);
              else
                newEyeDist = DTOPOINT(pntpos[0],pntpos[1],pntpos[2]);

              if (currEyeDist < 0.0 || newEyeDist < currEyeDist)
              {
                currEyeDist = newEyeDist;
                tag = currTag;
                if (unitcell)
                {
                  unitcell[0] = pbcCells[3 * pbcimg];
                  unitcell[1] = pbcCells[3 * pbcimg + 1];
                  unitcell[2] = pbcCells[3 * pbcimg + 2];
                }
              }
            }
            break;

          case DPICKPOINT_IARRAY:
            // loop over all of the pick points in the pick point index array
            DispCmdPickPointIndexArray *cmd =
                (DispCmdPickPointIndexArray *) cmdptr;
            float *pickpos;
            int *indices;
            cmd->getpointers (indices);

            int i;
            for (i = 0; i < cmd->numpicks; i++)
            {
              if (cmd->allselected)
              {
                pickpos = dataBlock + i * 3;
                currTag = i;
              }
              else
              {
                pickpos = dataBlock + indices[i] * 3;
                currTag = indices[i];
              }

              (transMat.top ()).multpoint3d (pickpos, pntpos);

              // check if in picking region ... different for 2D and 3D
              if (dim == 2)
              {
                // convert the 3D world coordinate to 2D absolute screen coord
                abs_screen_loc_3D (pntpos, cpos);

                // check to see if the position falls in our picking region
                inRegion = (cpos[0] >= minX && cpos[0] <= maxX
                    && cpos[1] >= minY && cpos[1] <= maxY);
              }
              else
              {
                // just check to see if the position is in a box centered on our
                // pointer.  The pointer position should already be transformed.
                inRegion = (pntpos[0] >= fminX && pntpos[0] <= fmaxX
                    && pntpos[1] >= fminY && pntpos[1] <= fmaxY
                    && pntpos[2] >= fminZ && pntpos[2] <= fmaxZ);
              }

              // has a hit occurred?
              if (inRegion)
              {
                // yes, see if it is closer to the eye than earlier hits
                if (dim == 2)
                  newEyeDist = DTOEYE(pntpos[0], pntpos[1], pntpos[2]);
                else
                  newEyeDist = DTOPOINT(pntpos[0],pntpos[1],pntpos[2]);

                if (currEyeDist < 0.0 || newEyeDist < currEyeDist)
                {
                  currEyeDist = newEyeDist;
                  tag = currTag;
                  if (unitcell)
                  {
                    unitcell[0] = pbcCells[3 * pbcimg];
                    unitcell[1] = pbcCells[3 * pbcimg + 1];
                    unitcell[2] = pbcCells[3 * pbcimg + 2];
                  }
                }
              }
            }
            break;
        }
      }

      // Pop the PBC image transform
      transMat.pop ();
    } // end of loop over PBC images

    // make sure we do not disturb the regular transformation matrix
    transMat.pop ();

    // return result; if tag >= 0, we found something
    eyedist = currEyeDist;
    return tag;
  }

