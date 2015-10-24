#ifndef WRAY_H__
#define WRAY_H__

__attribute__((target(mic)))
void wray(int np,int nr,int *line, float *weight, int *numb, float *snorm);

#endif