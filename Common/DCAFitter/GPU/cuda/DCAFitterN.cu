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
namespace o2::vertexing::device
{
namespace kernel
{
template <typename Fitter>
GPUg() void printKernel(Fitter* ft)
{
  if (threadIdx.x == 0) {
    printf(" =============== GPU DCA Fitter %d prongs ================\n", Fitter::getNProngs());
    ft->print();
    printf(" =========================================================\n");
  }
}

template <typename Fitter, typename... Tr>
GPUg() void processKernel(Fitter* ft, int* res, Tr*... tracks)
{
  *res = ft->process(*tracks...);
}
} // namespace kernel

/// CPU handlers
template <typename Fitter>
void print(const int nBlocks,
           const int nThreads,
           Fitter& ft)
{
  Fitter* ft_device;
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&ft_device), sizeof(Fitter)));
  gpuCheckError(cudaMemcpy(ft_device, &ft, sizeof(Fitter), cudaMemcpyHostToDevice));

  kernel::printKernel<<<nBlocks, nThreads>>>(ft_device);

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
}

template <typename Fitter, class... Tr>
int process(const int nBlocks,
            const int nThreads,
            Fitter& fitter,
            Tr&... args)
{
  Fitter* ft_device;
  std::array<o2::track::TrackParCov*, Fitter::getNProngs()> tracks_device;
  int result, *result_device;

  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&ft_device), sizeof(Fitter)));
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&result_device), sizeof(int)));

  int iArg{0};
  ([&] {
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&(tracks_device[iArg])), sizeof(o2::track::TrackParCov)));
    gpuCheckError(cudaMemcpy(tracks_device[iArg], &args, sizeof(o2::track::TrackParCov), cudaMemcpyHostToDevice));
    ++iArg;
  }(),
   ...);

  gpuCheckError(cudaMemcpy(ft_device, &fitter, sizeof(Fitter), cudaMemcpyHostToDevice));

  std::apply([&](auto&&... args) { kernel::processKernel<<<nBlocks, nThreads>>>(ft_device, result_device, args...); }, tracks_device);

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());

  gpuCheckError(cudaMemcpy(&result, result_device, sizeof(int), cudaMemcpyDeviceToHost));
  gpuCheckError(cudaMemcpy(&fitter, ft_device, sizeof(Fitter), cudaMemcpyDeviceToHost));
  iArg = 0;
  ([&] {
    gpuCheckError(cudaMemcpy(&args, tracks_device[iArg], sizeof(o2::track::TrackParCov), cudaMemcpyDeviceToHost));
    gpuCheckError(cudaFree(tracks_device[iArg]));
    ++iArg;
  }(),
   ...);

  gpuCheckError(cudaFree(result_device));

  return result;
}

template int process(const int, const int, o2::vertexing::DCAFitterN<2>&, o2::track::TrackParCov&, o2::track::TrackParCov&);
template int process(const int, const int, o2::vertexing::DCAFitterN<3>&, o2::track::TrackParCov&, o2::track::TrackParCov&, o2::track::TrackParCov&);
template void print(const int, const int, o2::vertexing::DCAFitterN<2>&);
template void print(const int, const int, o2::vertexing::DCAFitterN<3>&);
} // namespace o2::vertexing::device