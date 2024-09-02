// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef __HIPCC__
#include "hip/hip_runtime.h"
#else
#include <cuda.h>
#endif

#include "GPUCommonDef.h"
#include "DCAFitter/DCAFitterN.h"
// #include "MathUtils/SMatrixGPU.h"

#define gpuCheckError(x)                \
  {                                     \
    gpuAssert((x), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if (abort) {
      throw std::runtime_error("GPU assert failed.");
    }
  }
}
namespace o2::vertexing::gpu
{
namespace kernel
{
GPUg() void printKernel(o2::vertexing::DCAFitterN<2>* ft)
{
  if (threadIdx.x == 0) {
    printf(" =============== GPU DCA Fitter ================\n");
    ft->print();
    printf(" ===============================================\n");
  }
}

GPUg() void processKernel(o2::vertexing::DCAFitterN<2>* ft, o2::track::TrackParCov* t1, o2::track::TrackParCov* t2, int* res)
{
  *res = ft->process(*t1, *t2);
}
} // namespace kernel

} // namespace o2::vertexing::gpu
