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
/// \file Kernels.h
/// \author: mconcas@cern.ch

#ifndef GPU_BENCHMARK_KERNELS_H
#define GPU_BENCHMARK_KERNELS_H

#include "GPUCommonDef.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace o2
{
namespace benchmark
{
void printDevices();
void init();
template <typename... T>
float measure(void (*task)(T...), const char* taskName, T&&... args)
{
  float diff{0.f};

  auto start = std::chrono::high_resolution_clock::now();
  (*task)(std::forward<T>(args)...);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> diff_t{end - start};
  diff = diff_t.count();

  std::cout << std::setw(2) << " - " << taskName << " completed in: " << diff << " ms" << std::endl;

  return diff;
}

class GPUbenchmark final
{
 public:
  GPUbenchmark() = default;
  virtual ~GPUbenchmark() = default;
  void run();
};

// Steers
void GPUbenchmark::run()
{
  // printDevices();
  measure(&init, "Init");
}
} // namespace benchmark
} // namespace o2
#endif


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