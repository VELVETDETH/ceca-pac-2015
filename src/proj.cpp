#include <math.h>
#include <stdlib.h>

#include "wray.h"
#include "proj.h"
#include "proto.h"

void A(float *g,float *f) {
    int line[IMGSIZE*2+10];
    float weight[IMGSIZE*2+10];
    int np,nr,i,numb;
    float snorm;
    for (i = 0; i<NPROJ*NRAY; ++i) g[i] = 0;
    for (np = 0; np<NPROJ; ++np) {
        for (nr = 0; nr<NRAY; ++nr) {
            for (i = 0; i<IMGSIZE*2; ++i) {
                line[i] = 0;
                weight[i] = 0.;
            }
            wray(np,nr, line, weight, &numb, &snorm);
            for (i = 0; i<numb; ++i) {
                g[np*NRAY+nr] += f[line[i]]*weight[i];
            }
        }
    }
}

void AStar(float *f,float *g) {
    int line[IMGSIZE*2+10];
    float weight[IMGSIZE*2+10];
    int np,nr,i,numb;
    float snorm;
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) f[i] = 0;
    for (np = 0; np<NPROJ; ++np) {
        for (nr = 0; nr<NRAY; ++nr) {
            for (i = 0; i<IMGSIZE*2; ++i) {
                line[i] = 0;
                weight[i] = 0.;
            }
            wray(np,nr, line, weight, &numb, &snorm);
            for (i = 0; i<numb; ++i) {
                f[line[i]] += g[np*NRAY+nr]*weight[i];
            }
        }
    }
}

float bpseudo(float* rec, int np, int nr, int* line, float* weight, int * numb, float* snorm) {
    int i;
    float sum = 0.0;

    wray(np, nr, line, weight, numb, snorm);

    if (*numb != 0) {
        for (i = 0; i < *numb; ++i) {
            sum += rec[line[i]] * weight[i];
        }
    }
    return sum;
}

void pick(int* np, int* nr) {
    /*
    *np = rand()%NPROJ;
    *nr = rand()%NRAY;
    */
	++(*nr);
	if (*nr >= NRAY) {
		++(*np);
		*nr = 0;
	}
	if (*np >= NPROJ) *np = 0;

}