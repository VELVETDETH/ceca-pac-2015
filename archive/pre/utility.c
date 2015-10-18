#include <stdio.h>
#include <immintrin.h>
#include "proto.h"

float sqr(float x) { return x*x; }

void read_phantom(float *img, const char *file_name) {
    FILE *fin = fopen(file_name,"r");
   
    int i;
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i)
        fscanf(fin,"%f",img+i);
    fclose(fin);
}

void normalize(float *img) {
    float minval = 1e30,maxval = -1e30;
    int i;
//#pragma omp parallel for num_threads(120)
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        float tmp = img[i];
        if (minval>tmp) minval = tmp;
        if (maxval<tmp) maxval = tmp;
    }
    maxval -= minval;
//#pragma omp parallel for num_threads(120)
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i)
        img[i] = (img[i]-minval)/maxval*255.;
}

void write_file(float *img, const char *file_name) {
    FILE *fout = fopen(file_name,"w");
    int i,j;
    for (i = 0; i<IMGSIZE; ++i) {
        for (j = 0; j<IMGSIZE; ++j)
            fprintf(fout,"%f ",img[i*IMGSIZE+j]);
        fprintf(fout,"\n");
    }
    fclose(fout);
}

// void ini_const_m512(){
// 
//     mIMGSIZE = _mm512_set1_epi32(IMGSIZE);
//     mNPROJ = _mm512_set1_epi32(NPROJ);
//     mNRAY = _mm512_set1_epi32(NRAY);
//     numi1 = _mm512_set1_epi32(1);
//     numi0 = _mm512_set1_epi32(0);
//     numiIMG2 = _mm512_set1_epi32(IMGSIZE*IMGSIZE);
//     numiIMG1 = _mm512_set1_epi32(IMGSIZE*IMGSIZE-1);
//     numiIMG = _mm512_set1_epi32(IMGSIZE);
//     numf4 = _mm512_set1_ps(4.0);
//     numf0 = _mm512_set1_ps(0.0);
//     numf255 = _mm512_set1_ps(255.0);
//     mPI = _mm512_set1_ps(PI);
//     mPIdivNP = _mm512_set1_ps(PI/NPROJ);
//     mALPHA = _mm512_set1_ps(ALPHA);
//     mBETA = _mm512_set1_ps(BETA);
//     mEPSILON = _mm512_set1_ps(EPSILON);
// 
// 
// }
