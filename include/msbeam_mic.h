#ifndef MSBEAM_MIC_H__
#define MSBEAM_MIC_H__
  
#include "msbeam.h"

class MSBeamMic: public MSBeamBase {
public:
  void msbeam(float *f, float *v, float *g, int num_thread);
};

#endif