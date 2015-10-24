#ifndef MSBEAM_OFFLOAD_CPU_H__
#define MSBEAM_OFFLOAD_CPU_H__

// base class
#include "msbeam.h"

class MSBeamOffloadCpu: public MSBeamBase {
public:
  void msbeam(float *f, float *v, float *g, int num_thread);
};

#endif