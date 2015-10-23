#include <stdio.h>
#include <stdlib.h>

#include "proto.h"

float sqr(float x) { return x*x; }

void read_phantom(float *img, const char *file_name) {
    FILE *fin = fopen(file_name,"r");
    if (!fin) {
        fprintf(stderr, "Can't open file %s\n", file_name);
        exit(1);
    }
    int i;
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i)
        fscanf(fin,"%f",img+i);
    fclose(fin);
}

void normalize(float *img) {
    float minval = 1e30,maxval = -1e30;
    int i;
    for (i = 0; i < IMGSIZE * IMGSIZE; ++i) {
        float tmp = img[i];
        if (minval>tmp) minval = tmp;
        if (maxval<tmp) maxval = tmp;
    }
    maxval -= minval;
    for (i = 0; i < IMGSIZE * IMGSIZE; ++i)
        img[i] = (img[i]-minval)/maxval*255.;
}

void write_file(float *img, const char *file_name) {
    FILE *fout = fopen(file_name,"w");
    int i,j;
    for (i = 0; i < IMGSIZE; ++i) {
        for (j = 0; j < IMGSIZE; ++j)
            fprintf(fout,"%f ", img[i*IMGSIZE+j]);
        fprintf(fout,"\n");
    }
    fclose(fout);
}