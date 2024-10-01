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

#include <numeric>

#include "GPUCommonDef.h"
#include "DCAFitter/DCAFitterN.h"
#include "DeviceInterface/GPUInterface.h"

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
GPUg() void processBatchKernel(Fitter* fitters, int* results, size_t off, size_t N, Tr*... tracks)
{
  for (auto iThread{blockIdx.x * blockDim.x + threadIdx.x}; iThread < N; iThread += blockDim.x * gridDim.x) {
    results[iThread + off] = fitters[iThread + off].process(tracks[iThread + off]...);
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
void processBulk(const int nBlocks,
                 const int nThreads,
                 const int nBatches,
                 std::vector<Fitter>& fitters,
                 std::vector<int>& results,
                 std::vector<Tr>&... args)
{
  auto* gpuInterface = GPUInterface::Instance();
  kernel::warmUpGpuKernel<<<1, 1, 0, gpuInterface->getNextStream()>>>();

  // Benchmarking events
  std::vector<float> ioUp(nBatches), ioDown(nBatches), kerElapsed(nBatches);
  std::vector<cudaEvent_t> startIOUp(nBatches), endIOUp(nBatches), startIODown(nBatches), endIODown(nBatches), startKer(nBatches), endKer(nBatches);
  for (int iBatch{0}; iBatch < nBatches; ++iBatch) {
    gpuCheckError(cudaEventCreate(&startIOUp[iBatch]));
    gpuCheckError(cudaEventCreate(&endIOUp[iBatch]));
    gpuCheckError(cudaEventCreate(&startIODown[iBatch]));
    gpuCheckError(cudaEventCreate(&endIODown[iBatch]));
    gpuCheckError(cudaEventCreate(&startKer[iBatch]));
    gpuCheckError(cudaEventCreate(&endKer[iBatch]));
  }

  // Tracks
  std::array<o2::track::TrackParCov*, Fitter::getNProngs()> tracks_device;
  int iArg{0};
  ([&] {
    gpuInterface->registerBuffer(reinterpret_cast<void*>(args.data()), sizeof(Tr) * args.size());
    gpuInterface->allocDevice(reinterpret_cast<void**>(&(tracks_device[iArg])), sizeof(Tr) * args.size());
    ++iArg;
  }(),
   ...);

  // Fitters
  gpuInterface->registerBuffer(reinterpret_cast<void*>(fitters.data()), sizeof(Fitter) * fitters.size());
  Fitter* fitters_device;
  gpuInterface->allocDevice(reinterpret_cast<void**>(&fitters_device), sizeof(Fitter) * fitters.size());

  // Results
  gpuInterface->registerBuffer(reinterpret_cast<void*>(results.data()), sizeof(int) * fitters.size());
  int* results_device;
  gpuInterface->allocDevice(reinterpret_cast<void**>(&results_device), sizeof(int) * fitters.size());

  // R.R. Computation
  int totalSize = fitters.size();
  int batchSize = totalSize / nBatches;
  int remainder = totalSize % nBatches;

  for (int iBatch{0}; iBatch < nBatches; ++iBatch) {
    auto& stream = gpuInterface->getNextStream();
    auto offset = iBatch * batchSize + std::min(iBatch, remainder);
    auto nFits = batchSize + (iBatch < remainder ? 1 : 0);

    gpuCheckError(cudaEventRecord(startIOUp[iBatch], stream));
    gpuCheckError(cudaMemcpyAsync(fitters_device + offset, fitters.data() + offset, sizeof(Fitter) * nFits, cudaMemcpyHostToDevice, stream));
    iArg = 0;
    ([&] {
      gpuCheckError(cudaMemcpyAsync(tracks_device[iArg] + offset, args.data() + offset, sizeof(Tr) * nFits, cudaMemcpyHostToDevice, stream));
      ++iArg;
    }(),
     ...);
    gpuCheckError(cudaEventRecord(endIOUp[iBatch], stream));

    gpuCheckError(cudaEventRecord(startKer[iBatch], stream));
    std::apply([&](auto&&... args) { kernel::processBatchKernel<<<nBlocks, nThreads, 0, stream>>>(fitters_device, results_device, offset, nFits, args...); }, tracks_device);
    gpuCheckError(cudaEventRecord(endKer[iBatch], stream));

    gpuCheckError(cudaPeekAtLastError());
    iArg = 0;
    gpuCheckError(cudaEventRecord(startIODown[iBatch], stream));
    ([&] {
      gpuCheckError(cudaMemcpyAsync(args.data() + offset, tracks_device[iArg] + offset, sizeof(Tr) * nFits, cudaMemcpyDeviceToHost, stream));
      ++iArg;
    }(),
     ...);

    gpuCheckError(cudaMemcpyAsync(fitters.data() + offset, fitters_device + offset, sizeof(Fitter) * nFits, cudaMemcpyDeviceToHost, stream));
    gpuCheckError(cudaMemcpyAsync(results.data() + offset, results_device + offset, sizeof(int) * nFits, cudaMemcpyDeviceToHost, stream));
    gpuCheckError(cudaEventRecord(endIODown[iBatch], stream));
  }

  ([&] { gpuInterface->unregisterBuffer(args.data()); }(), ...);

  for (auto* tracksD : tracks_device) {
    gpuInterface->freeDevice(tracksD);
  }

  gpuInterface->freeDevice(fitters_device);
  gpuInterface->freeDevice(results_device);
  gpuInterface->unregisterBuffer(fitters.data());
  gpuInterface->unregisterBuffer(results.data());

  // Do benchmarks
  gpuCheckError(cudaDeviceSynchronize());
  for (int iBatch{0}; iBatch < nBatches; ++iBatch) {
    gpuCheckError(cudaEventElapsedTime(&ioUp[iBatch], startIOUp[iBatch], endIOUp[iBatch]));
    gpuCheckError(cudaEventElapsedTime(&kerElapsed[iBatch], startKer[iBatch], endKer[iBatch]));
    gpuCheckError(cudaEventElapsedTime(&ioDown[iBatch], startIODown[iBatch], endIODown[iBatch]));
  }

  float totalUp = std::accumulate(ioUp.begin(), ioUp.end(), 0.f);
  float totalDown = std::accumulate(ioDown.begin(), ioDown.end(), 0.f);
  float totalKernels = std::accumulate(kerElapsed.begin(), kerElapsed.end(), 0.f);
  LOGP(info, "Config: {} batches, {} blocks, {} threads", nBatches, nBlocks, nThreads);
  LOGP(info, "Total I/O time: Up {} ms Avg {} ms, Down {} ms Avg {} ms", totalUp, totalUp / float(nBatches), totalDown, totalDown / (float)nBatches);
  LOGP(info, "Total Kernel time: {} ms Avg {} ms", totalKernels, totalKernels / (float)nBatches);

  for (int iBatch{0}; iBatch < nBatches; ++iBatch) {
    gpuCheckError(cudaEventDestroy(startIOUp[iBatch]));
    gpuCheckError(cudaEventDestroy(endIOUp[iBatch]));
    gpuCheckError(cudaEventDestroy(startIODown[iBatch]));
    gpuCheckError(cudaEventDestroy(endIODown[iBatch]));
    gpuCheckError(cudaEventDestroy(startKer[iBatch]));
    gpuCheckError(cudaEventDestroy(endKer[iBatch]));
  }
}

template void processBulk(const int,
                          const int,
                          const int,
                          std::vector<o2::vertexing::DCAFitterN<2>>&,
                          std::vector<int>&,
                          std::vector<o2::track::TrackParCov>&,
                          std::vector<o2::track::TrackParCov>&);
template void processBulk(const int,
                          const int,
                          const int,
                          std::vector<o2::vertexing::DCAFitterN<3>>&,
                          std::vector<int>&,
                          std::vector<o2::track::TrackParCov>&,
                          std::vector<o2::track::TrackParCov>&,
                          std::vector<o2::track::TrackParCov>&);
template int process(const int, const int, o2::vertexing::DCAFitterN<2>&, o2::track::TrackParCov&, o2::track::TrackParCov&);
template int process(const int, const int, o2::vertexing::DCAFitterN<3>&, o2::track::TrackParCov&, o2::track::TrackParCov&, o2::track::TrackParCov&);
template void print(const int, const int, o2::vertexing::DCAFitterN<2>&);
template void print(const int, const int, o2::vertexing::DCAFitterN<3>&);
} // namespace o2::vertexing::device