
#include <cmath>
#include "proto.h"
#include "wray.h"

__attribute__((target(mic)))
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

__attribute__((target(mic)))
void wray(int np,int nr,int *line, float *weight, int *numb, float *snorm) {
    float sorx, sory, cf, sf, rcosphi, tanphi, rtnphi, c, fp, w, yc;
    int ix, iy, incx, incy, index, next, flip, flipxy;
        
    *numb = 0;
    *snorm = 0;
    
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

    for(;;) {
        switch(next) {
        case 1:
            if (iy+1>= IMGSIZE || ix<0) return;

            w = rcosphi * (1 - yc * rtnphi );
            weight[*numb] = w;
            line[*numb] = index;
            ++(*numb);
            (*snorm) += w * w;
            iy ++;
            index += incy;
            
            if (ix+1>=IMGSIZE) return;

            w = rcosphi * ( yc * rtnphi );
            weight[*numb] = w;
            line[*numb] = index;
            ++(*numb);
            (*snorm) += w * w;
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
            (*snorm) += w * w;
            ix ++;
            index += incx;

            yc += tanphi;
            if (yc>=0) next = 1;
            else next = 2;
            break;
        }
    }
}