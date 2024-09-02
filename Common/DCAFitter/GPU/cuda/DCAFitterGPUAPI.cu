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

#include "DCAFitter/DCAFitterN.h"

#include "ReconstructionDataFormats/Track.h"

#include <iostream>
#include <cstdint>

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
/// Print fitter
void doPrintOnDevice(o2::vertexing::DCAFitterN<2>* ft)
{
  DCAFitterN<2>* ft_device;
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&ft_device), sizeof(o2::vertexing::DCAFitterN<2>)));
  gpuCheckError(cudaMemcpy(ft_device, ft, sizeof(o2::vertexing::DCAFitterN<2>), cudaMemcpyHostToDevice));

  gpu::kernel::printKernel<<<1, 1>>>(ft_device);

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());

  gpuCheckError(cudaFree(ft_device));
}

/// Call the process(track, ...) method
int doProcessOnDevice(o2::vertexing::DCAFitterN<2>* ft, o2::track::TrackParCov* t1, o2::track::TrackParCov* t2)
{
  DCAFitterN<2>* ft_device;
  o2::track::TrackParCov* t1_device;
  o2::track::TrackParCov* t2_device;
  int result, *result_device;

  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&ft_device), sizeof(o2::vertexing::DCAFitterN<2>)));
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&t1_device), sizeof(o2::track::TrackParCov)));
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&t2_device), sizeof(o2::track::TrackParCov)));
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&result_device), sizeof(int)));

  gpuCheckError(cudaMemcpy(ft_device, ft, sizeof(o2::vertexing::DCAFitterN<2>), cudaMemcpyHostToDevice));
  gpuCheckError(cudaMemcpy(t1_device, t1, sizeof(o2::track::TrackParCov), cudaMemcpyHostToDevice));
  gpuCheckError(cudaMemcpy(t2_device, t2, sizeof(o2::track::TrackParCov), cudaMemcpyHostToDevice));

  gpu::kernel::processKernel<<<1, 1>>>(ft_device, t1_device, t2_device, result_device);
  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());

  gpuCheckError(cudaMemcpy(&result, result_device, sizeof(int), cudaMemcpyDeviceToHost));
  gpuCheckError(cudaFree(ft_device));
  gpuCheckError(cudaFree(t1_device));
  gpuCheckError(cudaFree(t2_device));
  gpuCheckError(cudaFree(result_device));

  return result;
}
} // namespace o2::vertexing::gpu