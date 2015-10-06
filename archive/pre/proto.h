
#ifndef _PROTO_H_
#define _PROTO_H_

#include <immintrin.h>
#define IMGSIZE 512
#define NPROJ 180
#define NRAY 768

#define ALPHA 20
#define BETA 0.35
#define EPSILON 0.001

#define ALL_ITER 10

#define F_TOL 1e-8
#define GRID_TOL 1e-2

#define PI 3.14159265359

__m512i mIMGSIZE;
__m512i mNPROJ;
__m512i mNRAY;

__m512i numi1;
__m512i numi0;
__m512i numiIMG2,numiIMG1;
__m512i numiIMG;
__m512 numf4;
__m512 numf0;
__m512 numf255;
__m512 mPI;
__m512 mPIdivNP;
__m512 mALPHA;
__m512 mBETA;
__m512 mEPSILON;



typedef __attribute__((aligned(64))) union zmmi{
	__m512i reg;
	int elems[16];
} zmmi_t;

typedef __attribute__((aligned(64))) union zmmf{
	__m512 reg;
	float elems[16];
} zmmf_t;

typedef __attribute__((aligned(2))) union zmmb{
	__mmask16 reg;
	unsigned short elems;
} zmmb_t;


#endif
