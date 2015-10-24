#ifndef _UTILITY_H_
#define _UTILITY_H_

#include "proto.h"

#define sqr(x) ((x) * (x))

void read_phantom(float *img, const char *file_name);

void write_file(float *img, const char *file_name);

void normalize(float *img);

#endif