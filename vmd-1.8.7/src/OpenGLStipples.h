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
 *      $RCSfile: OpenGLStipples.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $       $Date: 2009/04/29 15:43:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Stipple-based transparency rendering patterns
 ***************************************************************************/

// does this material have transparency?
// If so enable transparency via one of these methods:
//  A) alpha blending with two-pass rendering and painter's algorithm
//  B) alpha blending with two-pass rendering and alpha test
//  C) screen-door transparency with polygon stipple trick
//  Removed since we don't need to worry about changing transparency
//  every triangle - it's done once per command list now (or thereabouts).

// polygon stipple pattern
static GLubyte eighthtone[] = {
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00,
 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
 0x10, 0x10, 0x10, 0x10, 0x00, 0x00, 0x00, 0x00
};

static GLubyte quartertone[] = {
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44,
 0x11, 0x11, 0x11, 0x11, 0x44, 0x44, 0x44, 0x44
};

static GLubyte halftone[] = {
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55,
 0xAA, 0xAA, 0xAA, 0xAA, 0x55, 0x55, 0x55, 0x55
};

static GLubyte threequartertone[] = {
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD,
 0x77, 0x77, 0x77, 0x77, 0xDD, 0xDD, 0xDD, 0xDD
};

static GLubyte seveneighthtone[] = {
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xEF, 0xEF, 0xEF, 0xEF
};

static GLubyte ninesixteentone[] = {
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF,
 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
 0xEF, 0xEF, 0xEF, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF
};


