#ifndef _PROJ_H_
#define _PROJ_H_

#include "proto.h"

void pick(int* np, int* nr);
void A(float *g,float *f);
void AStar(float *f,float *g);
void wray(int np,int nr,int *line, float *weight, int *numb);
//float bpseudo(float* rec, int np, int nr, int* line, float* weight, int * numb, float* snorm);

#endif
