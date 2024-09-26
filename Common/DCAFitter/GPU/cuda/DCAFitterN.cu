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
GPUg() void warmUpGpuKernel()
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

template <typename Fitter>
GPUg() void printKernel(Fitter* fitter)
{
  if (threadIdx.x == 0) {
    printf(" =============== GPU DCA Fitter %d prongs =================\n", Fitter::getNProngs());
    fitter->print();
    printf(" =========================================================\n");
  }
}

template <typename Fitter, typename... Tr>
GPUg() void processKernel(Fitter* fitter, int* res, Tr*... tracks)
{
  *res = fitter->process(*tracks...);
}

template <typename Fitter, typename... Tr>
GPUg() void processBulkKernel(Fitter* fitters, int* results, unsigned int N, Tr*... tracks)
{
  for (auto iThread{blockIdx.x * blockDim.x + threadIdx.x}; iThread < N; iThread += blockDim.x * gridDim.x) {
    results[iThread] = fitters[iThread].process(tracks[iThread]...);
  }
}

} // namespace kernel

/// CPU handlers
template <typename Fitter>
void print(const int nBlocks,
           const int nThreads,
           Fitter& fitter)
{
  Fitter* fitter_device;
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&fitter_device), sizeof(Fitter)));
  gpuCheckError(cudaMemcpy(fitter_device, &fitter, sizeof(Fitter), cudaMemcpyHostToDevice));

  kernel::printKernel<<<nBlocks, nThreads>>>(fitter_device);

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());
}

template <typename Fitter, class... Tr>
int process(const int nBlocks,
            const int nThreads,
            Fitter& fitter,
            Tr&... args)
{
  Fitter* fitter_device;
  std::array<o2::track::TrackParCov*, Fitter::getNProngs()> tracks_device;
  int result, *result_device;

  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&fitter_device), sizeof(Fitter)));
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&result_device), sizeof(int)));

  int iArg{0};
  ([&] {
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&(tracks_device[iArg])), sizeof(o2::track::TrackParCov)));
    gpuCheckError(cudaMemcpy(tracks_device[iArg], &args, sizeof(o2::track::TrackParCov), cudaMemcpyHostToDevice));
    ++iArg;
  }(),
   ...);

  gpuCheckError(cudaMemcpy(fitter_device, &fitter, sizeof(Fitter), cudaMemcpyHostToDevice));

  std::apply([&](auto&&... args) { kernel::processKernel<<<nBlocks, nThreads>>>(fitter_device, result_device, args...); }, tracks_device);

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());

  gpuCheckError(cudaMemcpy(&result, result_device, sizeof(int), cudaMemcpyDeviceToHost));
  gpuCheckError(cudaMemcpy(&fitter, fitter_device, sizeof(Fitter), cudaMemcpyDeviceToHost));
  iArg = 0;
  ([&] {
    gpuCheckError(cudaMemcpy(&args, tracks_device[iArg], sizeof(o2::track::TrackParCov), cudaMemcpyDeviceToHost));
    gpuCheckError(cudaFree(tracks_device[iArg]));
    ++iArg;
  }(),
   ...);

  gpuCheckError(cudaFree(fitter_device));
  gpuCheckError(cudaFree(result_device));

  return result;
}

template <typename Fitter, class... Tr>
std::vector<int> processBulk(const int nBlocks,
                             const int nThreads,
                             std::vector<Fitter>& fitters,
                             std::vector<Tr>&... args)
{
  kernel::warmUpGpuKernel<<<1, 1>>>();

  cudaEvent_t start, stop;
  gpuCheckError(cudaEventCreate(&start));
  gpuCheckError(cudaEventCreate(&stop));
  const auto nFits{fitters.size()}; // for clarity: size of all the vectors needs to be equal, not enforcing it here yet.
  std::vector<int> results(nFits);
  int* results_device;
  Fitter* fitters_device;
  std::array<o2::track::TrackParCov*, Fitter::getNProngs()> tracks_device;

  int iArg{0};
  ([&] {
    gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&(tracks_device[iArg])), sizeof(Tr) * args.size()));
    gpuCheckError(cudaMemcpy(tracks_device[iArg], args.data(), sizeof(Tr) * args.size(), cudaMemcpyHostToDevice));
    ++iArg;
  }(),
   ...);
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&results_device), sizeof(int) * nFits));
  gpuCheckError(cudaMalloc(reinterpret_cast<void**>(&fitters_device), sizeof(Fitter) * nFits));
  gpuCheckError(cudaMemcpy(fitters_device, fitters.data(), sizeof(Fitter) * nFits, cudaMemcpyHostToDevice));

  gpuCheckError(cudaEventRecord(start));
  std::apply([&](auto&&... args) { kernel::processBulkKernel<<<nBlocks, nThreads>>>(fitters_device, results_device, nFits, args...); }, tracks_device);
  gpuCheckError(cudaEventRecord(stop));

  gpuCheckError(cudaPeekAtLastError());
  gpuCheckError(cudaDeviceSynchronize());

  gpuCheckError(cudaMemcpy(results.data(), results_device, sizeof(int) * results.size(), cudaMemcpyDeviceToHost));
  gpuCheckError(cudaMemcpy(fitters.data(), fitters_device, sizeof(Fitter) * nFits, cudaMemcpyDeviceToHost));

  iArg = 0;
  ([&] {
    gpuCheckError(cudaMemcpy(args.data(), tracks_device[iArg], sizeof(Tr) * args.size(), cudaMemcpyDeviceToHost));
    gpuCheckError(cudaFree(tracks_device[iArg]));
    ++iArg;
  }(),
   ...);

  gpuCheckError(cudaFree(fitters_device));
  gpuCheckError(cudaFree(results_device));
  gpuCheckError(cudaEventSynchronize(stop));

  float milliseconds = 0;
  gpuCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

  LOGP(info, "Kernel run in: {} ms using {} blocks and {} threads.", milliseconds, nBlocks, nThreads);
  return results;
}

template std::vector<int> processBulk(const int, const int, std::vector<o2::vertexing::DCAFitterN<2>>&, std::vector<o2::track::TrackParCov>&, std::vector<o2::track::TrackParCov>&);
template std::vector<int> processBulk(const int, const int, std::vector<o2::vertexing::DCAFitterN<3>>&, std::vector<o2::track::TrackParCov>&, std::vector<o2::track::TrackParCov>&, std::vector<o2::track::TrackParCov>&);
template int process(const int, const int, o2::vertexing::DCAFitterN<2>&, o2::track::TrackParCov&, o2::track::TrackParCov&);
template int process(const int, const int, o2::vertexing::DCAFitterN<3>&, o2::track::TrackParCov&, o2::track::TrackParCov&, o2::track::TrackParCov&);
template void print(const int, const int, o2::vertexing::DCAFitterN<2>&);
template void print(const int, const int, o2::vertexing::DCAFitterN<3>&);
} // namespace o2::vertexing::device