
#ifndef __Hershey_h__
#define __Hershey_h__

/* private handle data structure */
typedef struct {
  float lm;
  float rm;
  const char *p;
} hersheyhandle;

extern char* hersheyFontData[];
void hersheyDrawLetter(const char* cp);
void hersheyDrawInitLetter(hersheyhandle *hh, const char ch, 
                           float *lm, float *rm);
int hersheyDrawNextLine(hersheyhandle *hh, int *draw, float *x, float *y);

#endif /* __Hershey_h__ */
