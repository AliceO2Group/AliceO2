// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionCUDAInternals.h
/// \author David Rohr

// All CUDA-header related stuff goes here, so we can run CING over GPUReconstructionCUDA

#ifndef GPURECONSTRUCTIONCUDAINTERNALS_H
#define GPURECONSTRUCTIONCUDAINTERNALS_H

#include "GPULogging.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUReconstructionCUDAInternals {
  CUcontext CudaContext;                       // Pointer to CUDA context
  cudaStream_t Streams[GPUCA_MAX_STREAMS];     // Pointer to array of CUDA Streams
};

#define GPUFailedMsg(x) GPUFailedMsgA(x, __FILE__, __LINE__)
#define GPUFailedMsgI(x) GPUFailedMsgAI(x, __FILE__, __LINE__)

static int GPUFailedMsgAI(const long long int error, const char* file, int line)
{
  // Check for CUDA Error and in the case of an error display the corresponding error string
  if (error == cudaSuccess) {
    return (0);
  }
  GPUError("CUDA Error: %lld / %s (%s:%d)", error, cudaGetErrorString((cudaError_t)error), file, line);
  return 1;
}

static void GPUFailedMsgA(const long long int error, const char* file, int line)
{
  if (GPUFailedMsgAI(error, file, line)) {
    throw std::runtime_error("CUDA Failure");
  }
}

static_assert(std::is_convertible<cudaEvent_t, void*>::value, "CUDA event type incompatible to deviceEvent");

class ThrustVolatileAllocator
{
 public:
  typedef char value_type;

  ThrustVolatileAllocator(GPUReconstruction* r) : mRec(r) {}
  char* allocate(std::ptrdiff_t n) { return (char*)mRec->AllocateVolatileDeviceMemory(n); }

  void deallocate(char* ptr, size_t) {}

 private:
  GPUReconstruction* mRec;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

// Override synchronize call at end of thrust algorithm running on stream, just don't run cudaStreamSynchronize
THRUST_BEGIN_NS
namespace cuda_cub
{
typedef thrust::cuda_cub::execution_policy<thrust::cuda_cub::execute_on_stream> thrustStreamPolicy;
template <>
__host__ __device__ inline cudaError_t synchronize<thrustStreamPolicy>(thrustStreamPolicy& policy)
{
  return cudaSuccess;
}

} // namespace cuda_cub
THRUST_END_NS

#endif
