#ifndef _UTILITY_H_
#define _UTILITY_H_

#include "proto.h"

float sqr(float x);

void read_phantom(float *img, const char *file_name);

void write_file(float *img, const char *file_name);

void normalize(float *img);

void ini_const_m512();
#endif
