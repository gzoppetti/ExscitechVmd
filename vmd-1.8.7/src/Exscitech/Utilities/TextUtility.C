#include <GL/glut.h>

#include <stdio.h>
#include <string.h>
#include "Exscitech/Utilities/TextUtility.hpp"

void PrintText(int x, int y, const char* const string)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    gluOrtho2D(viewport[0], viewport[2], viewport[1], viewport[3]);
    //gluOrtho2D(0, 800, 0, 600);
    int len, i;
    glRasterPos2f(x, y);
    len = (int) strlen(string);
    for (i = 0; i < len; i++)
    {
      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
    }


    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
  }



