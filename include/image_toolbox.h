/**
 * include/image_toolbox.h:
 * Several functions for you to test the quality of the result images.
 * Function list:
 * 1. SSIM: Structural SIMilarity index.
 * 2. PNSR: Peak Signal-to-Notice Ratio
 * 3. MSE:  Mean Square Error
 *
 * Author: Vincent Zhao <vincentzhaorz@gmail.com>
 * Date: 2015.10.22
 */

#ifndef IMAGE_TOOLBOX_H__
#define IMAGE_TOOLBOX_H__

namespace ImageToolbox {
  // We assume all the indices returned are in double format
  // SSIM(x,y): for the pixel at (x, y), calculate the window with size m * n.
  // I and J are 2 input images
  // 
  // You could find the formula that I'm using on 
  //  https://en.wikipedia.org/wiki/Structural_similarity
  // WARNING: I have no idea whether m != n will affect the result as in the formula
  // the assumption is m = n. We may check this later.
  // 
  // You need to specify a local filter for SSIM calculation.
  double SSIM(int x, int y, float *I, float *J, int m, int n, double *filter);
  double MSSIM(float *I, float *J, int m, int n);
};

#endif