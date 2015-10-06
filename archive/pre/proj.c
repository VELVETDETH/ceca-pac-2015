#include <math.h>
#include <stdlib.h>

#include "proto.h"
#include "proj.h"
#define MIDRAY 384
#define MIDPIX 256
/*
void pick(int* np, int* nr);
void posit(int np,int nr,float *ax,float *ay,float *mx,float *my);
void wray(int np,int nr,int *line, float *weight, int *numb, float *snorm);
float bpseudo(float* rec, int np, int nr, int* line, float* weight, int * numb, float* snorm);
*/

void A(float *g,float *f) {
    int line[IMGSIZE*2+10];
    float weight[IMGSIZE*2+10];
    int np,nr,i,numb;
    for (i = 0; i<NPROJ*NRAY; ++i) g[i] = 0;
    for (np = 0; np<NPROJ; ++np) {
        for (nr = 0; nr<NRAY; ++nr) {
            for (i = 0; i<IMGSIZE*2; ++i) {
                line[i] = 0;
                weight[i] = 0.;
            }
            wray(np,nr, line, weight,&numb);
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
            wray(np,nr, line, weight, &numb);
            for (i = 0; i<numb; ++i) {
                f[line[i]] += g[np*NRAY+nr]*weight[i];
            }
        }
    }
}

void posit(int np,int nr,float *ax,float *ay,float *mx,float *my) {
    float sinth = sin(PI/NPROJ*np);
    float costh = cos(PI/NPROJ*np);
    
    int ind = nr-MIDRAY;
    int rwidth = 1; /* space between detectors */
    float tmp_length = (float)ind*rwidth;
    *ax = -sinth*tmp_length;
    *ay = costh*tmp_length;
    *mx = costh;
    *my = sinth;
}

void wray(int np,int nr,int *line, float *weight, int *numb) {
    float sorx, sory, cf, sf, rcosphi, tanphi, rtnphi, c, fp, w, yc;
    int ix, iy, incx, incy, index, next, flip, flipxy;
    


    int idx,idx_;
    int idy;
    float fy;    
    int tmpline[IMGSIZE][2]={0}; 
    float tmpweight[IMGSIZE][2]={0.0};
   
    float l; 

    *numb = 0;
    
    /* get the info of the line (np,nr) */
    posit(np, nr, &sorx, &sory, &cf, &sf);
    if (sf < -F_TOL) {
        sf = -sf;
        cf = -cf;
    }

    flip = 0;
    if (sf * cf < -F_TOL*F_TOL) {
        flip = 1;
        cf = -cf;
    }

    if (flip > 0) {
        sorx = -sorx;
    }

    if (fabs(sf) <= F_TOL) {
        if (sf < 0) sf = -F_TOL*F_TOL;
        else sf = F_TOL*F_TOL;
    }

    if (fabs(cf) <= F_TOL) {
        if (cf < 0) cf = -F_TOL*F_TOL;
        else cf = F_TOL*F_TOL;
    }

    if (fabs(cf) > F_TOL) tanphi = sf / cf;
    else tanphi = 1/(F_TOL*F_TOL);
    
    if (fabs(sf) > F_TOL) rtnphi = cf / sf;
    else rtnphi = 1/(F_TOL*F_TOL);

    flipxy = 0;
    if ( tanphi > 1+F_TOL ) {
        flipxy = 1;
        tanphi = 1/tanphi;
        rtnphi = 1/rtnphi;
        sf = sory;
        sory = sorx;
        sorx = sf;
    }

    rcosphi = ( sqrt( 1 + tanphi* tanphi));

    sory += IMGSIZE / 2.0;
    sorx += IMGSIZE / 2.0;
    c = sory - sorx * tanphi;

    if ((c - IMGSIZE) >= GRID_TOL ) return;
    if (c < -GRID_TOL) {
        c = -c * rtnphi;
        if((c - IMGSIZE) >= GRID_TOL) return;
        ix = (int) c;
        iy = 0;
        fp = ((float) (ix + 1)) - c;
        yc = fp * tanphi;
        yc += -1.0;
    }
    else {
        ix = 0;
        iy = (int) c;
        fp = ((float) (iy + 1)) - c;
        yc = -fp;
        yc += tanphi;
    }

    if (yc>=0) next = 1;
    else next = 2;


    /* calculate the initial index and the increment of index */
    

    if (flipxy > 0){
        index = iy + ix * IMGSIZE;
        incx = IMGSIZE;
        incy = 1;
        if (flip>0){ 
            index = IMGSIZE - 1 - iy + ix * IMGSIZE ; 
            incx = IMGSIZE;
            incy = -1;
        }
    }
    else {
        index = iy * IMGSIZE + ix;
        incx = 1;
        incy = IMGSIZE;
        if (flip>0) {
            index = IMGSIZE - 1 - ix + iy * IMGSIZE ;
            	incx = -1;
            incy = IMGSIZE;
        }
    }
/*   if(iy==0){
        iy=ix;ix=0;
	float tmpt;
	tmpt=tanphi;tanphi=rtnphi;rtnphi=tmpt;
	int tmpinc;
	tmpinc=incx;incx=incy;incy=tmpinc;
	rcosphi = ( sqrt( 1 + tanphi* tanphi));

    }
   for(idx=0;idx<IMGSIZE;++idx){
        fy=iy+(idx*tanphi);
        idy=(int)fy;
        if(idy+1>=IMGSIZE||idy<0) break;
	l=fy-idy+tanphi-1;
        if(l>0){
	    tmpweight[idx][0]=rcosphi * (1 - l * rtnphi );
	    tmpline[idx][0]  =idx*incx+idy*incy;
	    tmpweight[idx][1]=rcosphi * ( l * rtnphi );
	    tmpline[idx][1]  =idx*incx+(idy+1)*incy;
	}
	else{
	    tmpweight[idx][0]=rcosphi;
	    tmpline[idx][0]  =idx*incx+idy*incy;
	    tmpweight[idx][1]=0;

	}
    }
    for(idx_=0;idx_<idx;++idx_){
	weight[*numb]=tmpweight[idx_][0];
	line[*numb]  =tmpline[idx_][0];
	++(*numb);
	if(tmpweight[idx_][1]!=0){
	    weight[*numb]=tmpweight[idx_][1];
	    line[*numb]  =tmpline[idx_][1];
	    ++(*numb);
	}
    }
*/


    for(;;) {
        switch(next) {
        case 1:
            if (iy+1>= IMGSIZE || ix<0) return;

            w = rcosphi * (1 - yc * rtnphi );
            weight[*numb] = w;
            line[*numb] = index;
            ++(*numb);
            iy ++;
            index += incy;
            
            if (ix+1>=IMGSIZE) return;

            w = rcosphi * ( yc * rtnphi );
            weight[*numb] = w;
            line[*numb] = index;
            ++(*numb);
            ix ++;
            index += incx;

            yc += tanphi - 1.0;
            if (yc>=0) next = 1;
            else next = 2;
            break;
        case 2:
            if (iy>= IMGSIZE || ix+1>=IMGSIZE || ix<0) return;

            w = rcosphi;
            weight[*numb] = w;
            line[*numb] = index;
            ++(*numb);
            ix ++;
            index += incx;

            yc += tanphi;
            if (yc>=0) next = 1;
            else next = 2;
            break;
        }
    }

}
/*
float bpseudo(float* rec, int np, int nr, int* line, float* weight, int * numb, float* snorm) {
    int i;
    float sum = 0.0;

    wray(np, nr, line, weight, numb);

    if (*numb != 0) {
        for (i = 0; i < *numb; ++i) {
            sum += rec[line[i]] * weight[i];
        }
    }
    return sum;
}
*/
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
