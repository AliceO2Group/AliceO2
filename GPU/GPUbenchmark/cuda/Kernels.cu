// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Kernels.cu
/// \author: mconcas@cern.ch

#include <Kernels.h>
#include <iostream>
#include <iomanip>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define failed(...)                       \
  printf("%serror: ", KRED);              \
  printf(__VA_ARGS__);                    \
  printf("\n");                           \
  printf("error: TEST FAILED\n%s", KNRM); \
  exit(EXIT_FAILURE);

#define GPUCHECK(error)                                                                        \
  if (error != cudaSuccess) {                                                                  \
    printf("%serror: '%s'(%d) at %s:%d%s\n", KRED, cudaGetErrorString(error), error, __FILE__, \
           __LINE__, KNRM);                                                                    \
    failed("API returned error code.");                                                        \
  }

void printCompilerInfo()
{
#ifdef __NVCC__
  printf("compiler: nvcc\n");
#endif
}

double bytesToKB(size_t s) { return (double)s / (1024.0); }
double bytesToGB(size_t s) { return (double)s / (1024.0 * 1024.0 * 1024.0); }

#define printLimit(w1, limit, units)                                          \
  {                                                                           \
    size_t val;                                                               \
    cudaDeviceGetLimit(&val, limit);                                          \
    std::cout << setw(w1) << #limit ": " << val << " " << units << std::endl; \
  }

namespace o2
{
namespace benchmark
{
namespace gpu
{
GPUg() void helloKernel()
{
  printf("Hello World from GPU!\n");
}

} // namespace gpu
void printDeviceProp(int deviceId)
{
  using namespace std;
  const int w1 = 34;
  cout << left;
  cout << setw(w1)
       << "--------------------------------------------------------------------------------"
       << endl;
  cout << setw(w1) << "device#" << deviceId << endl;

  cudaDeviceProp props;
  GPUCHECK(cudaGetDeviceProperties(&props, deviceId));

  cout << setw(w1) << "Name: " << props.name << endl;
  cout << setw(w1) << "pciBusID: " << props.pciBusID << endl;
  cout << setw(w1) << "pciDeviceID: " << props.pciDeviceID << endl;
  cout << setw(w1) << "pciDomainID: " << props.pciDomainID << endl;
  cout << setw(w1) << "multiProcessorCount: " << props.multiProcessorCount << endl;
  cout << setw(w1) << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor
       << endl;
  cout << setw(w1) << "isMultiGpuBoard: " << props.isMultiGpuBoard << endl;
  cout << setw(w1) << "clockRate: " << (float)props.clockRate / 1000.0 << " Mhz" << endl;
  cout << setw(w1) << "memoryClockRate: " << (float)props.memoryClockRate / 1000.0 << " Mhz"
       << endl;
  cout << setw(w1) << "memoryBusWidth: " << props.memoryBusWidth << endl;
  cout << setw(w1) << "clockInstructionRate: " << (float)props.clockRate / 1000.0
       << " Mhz" << endl;
  cout << setw(w1) << "totalGlobalMem: " << fixed << setprecision(2)
       << bytesToGB(props.totalGlobalMem) << " GB" << endl;
#if !defined(__CUDACC__)
  cout << setw(w1) << "maxSharedMemoryPerMultiProcessor: " << fixed << setprecision(2)
       << bytesToKB(props.sharedMemPerMultiprocessor) << " KB" << endl;
#endif
#if defined(__HIPCC__)
  cout << setw(w1) << "maxSharedMemoryPerMultiProcessor: " << fixed << setprecision(2)
       << bytesToKB(props.maxSharedMemoryPerMultiProcessor) << " KB" << endl;
#endif
  cout << setw(w1) << "totalConstMem: " << props.totalConstMem << endl;
  cout << setw(w1) << "sharedMemPerBlock: " << (float)props.sharedMemPerBlock / 1024.0 << " KB"
       << endl;
  cout << setw(w1) << "canMapHostMemory: " << props.canMapHostMemory << endl;
  cout << setw(w1) << "regsPerBlock: " << props.regsPerBlock << endl;
  cout << setw(w1) << "warpSize: " << props.warpSize << endl;
  cout << setw(w1) << "l2CacheSize: " << props.l2CacheSize << endl;
  cout << setw(w1) << "computeMode: " << props.computeMode << endl;
  cout << setw(w1) << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << endl;
  cout << setw(w1) << "maxThreadsDim.x: " << props.maxThreadsDim[0] << endl;
  cout << setw(w1) << "maxThreadsDim.y: " << props.maxThreadsDim[1] << endl;
  cout << setw(w1) << "maxThreadsDim.z: " << props.maxThreadsDim[2] << endl;
  cout << setw(w1) << "maxGridSize.x: " << props.maxGridSize[0] << endl;
  cout << setw(w1) << "maxGridSize.y: " << props.maxGridSize[1] << endl;
  cout << setw(w1) << "maxGridSize.z: " << props.maxGridSize[2] << endl;
  cout << setw(w1) << "major: " << props.major << endl;
  cout << setw(w1) << "minor: " << props.minor << endl;
  cout << setw(w1) << "concurrentKernels: " << props.concurrentKernels << endl;
  cout << setw(w1) << "cooperativeLaunch: " << props.cooperativeLaunch << endl;
  cout << setw(w1) << "cooperativeMultiDeviceLaunch: " << props.cooperativeMultiDeviceLaunch << endl;
#if defined(__HIPCC__)
  cout << setw(w1) << "arch.hasGlobalInt32Atomics: " << props.arch.hasGlobalInt32Atomics << endl;
  cout << setw(w1) << "arch.hasGlobalFloatAtomicExch: " << props.arch.hasGlobalFloatAtomicExch
       << endl;
  cout << setw(w1) << "arch.hasSharedInt32Atomics: " << props.arch.hasSharedInt32Atomics << endl;
  cout << setw(w1) << "arch.hasSharedFloatAtomicExch: " << props.arch.hasSharedFloatAtomicExch
       << endl;
  cout << setw(w1) << "arch.hasFloatAtomicAdd: " << props.arch.hasFloatAtomicAdd << endl;
  cout << setw(w1) << "arch.hasGlobalInt64Atomics: " << props.arch.hasGlobalInt64Atomics << endl;
  cout << setw(w1) << "arch.hasSharedInt64Atomics: " << props.arch.hasSharedInt64Atomics << endl;
  cout << setw(w1) << "arch.hasDoubles: " << props.arch.hasDoubles << endl;
  cout << setw(w1) << "arch.hasWarpVote: " << props.arch.hasWarpVote << endl;
  cout << setw(w1) << "arch.hasWarpBallot: " << props.arch.hasWarpBallot << endl;
  cout << setw(w1) << "arch.hasWarpShuffle: " << props.arch.hasWarpShuffle << endl;
  cout << setw(w1) << "arch.hasFunnelShift: " << props.arch.hasFunnelShift << endl;
  cout << setw(w1) << "arch.hasThreadFenceSystem: " << props.arch.hasThreadFenceSystem << endl;
  cout << setw(w1) << "arch.hasSyncThreadsExt: " << props.arch.hasSyncThreadsExt << endl;
  cout << setw(w1) << "arch.hasSurfaceFuncs: " << props.arch.hasSurfaceFuncs << endl;
  cout << setw(w1) << "arch.has3dGrid: " << props.arch.has3dGrid << endl;
  cout << setw(w1) << "arch.hasDynamicParallelism: " << props.arch.hasDynamicParallelism << endl;
  cout << setw(w1) << "gcnArchName: " << props.gcnArchName << endl;
#endif
  cout << setw(w1) << "isIntegrated: " << props.integrated << endl;
  cout << setw(w1) << "maxTexture1D: " << props.maxTexture1D << endl;
  cout << setw(w1) << "maxTexture2D.width: " << props.maxTexture2D[0] << endl;
  cout << setw(w1) << "maxTexture2D.height: " << props.maxTexture2D[1] << endl;
  cout << setw(w1) << "maxTexture3D.width: " << props.maxTexture3D[0] << endl;
  cout << setw(w1) << "maxTexture3D.height: " << props.maxTexture3D[1] << endl;
  cout << setw(w1) << "maxTexture3D.depth: " << props.maxTexture3D[2] << endl;
#if defined(__HIPCC__)
  cout << setw(w1) << "isLargeBar: " << props.isLargeBar << endl;
  cout << setw(w1) << "asicRevision: " << props.asicRevision << endl;
#endif

  int deviceCnt;
  cudaGetDeviceCount(&deviceCnt);
  cout << setw(w1) << "peers: ";
  for (int i = 0; i < deviceCnt; i++) {
    int isPeer;
    cudaDeviceCanAccessPeer(&isPeer, i, deviceId);
    if (isPeer) {
      cout << "device#" << i << " ";
    }
  }
  cout << endl;
  cout << setw(w1) << "non-peers: ";
  for (int i = 0; i < deviceCnt; i++) {
    int isPeer;
    cudaDeviceCanAccessPeer(&isPeer, i, deviceId);
    if (!isPeer) {
      cout << "device#" << i << " ";
    }
  }
  cout << endl;

  size_t free, total;
  cudaMemGetInfo(&free, &total);

  cout << fixed << setprecision(2);
  cout << setw(w1) << "memInfo.total: " << bytesToGB(total) << " GB" << endl;
  cout << setw(w1) << "memInfo.free:  " << bytesToGB(free) << " GB (" << setprecision(0)
       << (float)free / total * 100.0 << "%)" << endl;
}

void hello_util()
{
  int deviceCnt;

  GPUCHECK(cudaGetDeviceCount(&deviceCnt));

  for (int i = 0; i < deviceCnt; i++) {
    cudaSetDevice(i);
    printDeviceProp(i);
  }

  // gpu::helloKernel<<<1, 1>>>();
  // displayCard();
}
} // namespace benchmark
} // namespace o2

/*In particular: I'd allocate one single large buffer filling almost the whole GPU memory, and then assume that it is more or less linear, at least if the GPU memory was free before.
I.e., at least the lower ~ 14 GB of the buffer should be in the lower 16 GB memory, and the higher ~14 GB in the upper 16 GP.

Then we partition this buffer in say 1GB segments, and run benchmarks in the segments individually, or in multiple segments in parallel.
For running on multiple segments in parallel, it would be interesting to split on the block level and on the thread level.
We should always start as many blocks as there are multiprocessors on the GPU, such that we have a 1 to 1 mapping without scheduling blocks.
We should make sure that the test runs long enough, say >5 seconds, then the initial scheduling should become irrelevant.

For the tests I want to run in the segments, I think these should be:
- Linear read in a multithreaded way: i.e. the standard GPU for loop:
for (int i = threadIdx.x; i < segmentSIze; i += blockDim.x) foo += array[i];
In the end we have to write foo to some output address to make sure the compiler cannot optimize anything.
- Then I'd do the same with some stride, i.e.:
foo += array[i * stride];
- I'd try a random access with some simple linear congruence RNG per thread to determine the address.
- Then I'd do the same with writing memory, and with copying memory.
- Finally the data type should be flexible, going from char to uint4.
That should cover most cases, but if you have more ideas, feel free to add something.*/