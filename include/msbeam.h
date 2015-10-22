#ifndef MSBEAM_H__
#define MSBEAM_H__

void min_image(float *f, float *v, float *g, int np, int nr, int *line, float *weight, int numb, float snorm);
void min_edge(float *f, float *v, int np,int nr,int *line,float *weight,int numb);
void min_wrapper(float *f, float *v, float *g, int np,int nr);
void msbeam(float *f, float *v, float *g);

#endif