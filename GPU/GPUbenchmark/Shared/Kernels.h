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
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

#define PARTITION_SIZE_GB 1
#define FREE_MEMORY_FRACTION_TO_ALLOCATE 0.95f
#define GB 1073741824

namespace o2
{
namespace benchmark
{

template <class T>
struct gpuState {
  int getMaxSegments()
  {
    return (double)scratchSize / (1024.0 * 1024.0 * 1024.0);
  }

  void computeScratchPtrs()
  {
    addresses.resize(getMaxSegments());
    for (size_t iBuffAddress{0}; iBuffAddress < getMaxSegments(); ++iBuffAddress) {
      addresses[iBuffAddress] = reinterpret_cast<T*>(reinterpret_cast<char*>(scratchPtr) + GB * iBuffAddress);
    }
  }

  static constexpr size_t getArraySize()
  {
    return static_cast<size_t>(GB * PARTITION_SIZE_GB / sizeof(T));
  }

  std::vector<T*> getScratchPtrs()
  {
    return addresses;
  }

  std::vector<std::vector<T>>& getHostBuffers()
  {
    return gpuBuffersHost;
  }

  // General containers and state
  T* scratchPtr;                              // Pointer to scratch buffer
  size_t scratchSize;                         // Size of scratch area (B)
  std::vector<T*> addresses;                  // Pointers to scratch partitions
  std::vector<std::vector<T>> gpuBuffersHost; // Host-based vector-ized data

  // Test-specific containers
  std::vector<T*> deviceReadingResultsPtrs; // Results of the reading test (single variable) on GPU
  std::vector<T> hostReadingResultsVector;  // Results of the reading test (single variable) on host

  // Configuration
  size_t nMaxThreadsPerDimension;

  // Static info
  size_t totalMemory;
  size_t nMultiprocessors;
  size_t nMaxThreadsPerBlock;
};

template <class buffer_type>
class GPUbenchmark final
{
 public:
  GPUbenchmark() = default;
  virtual ~GPUbenchmark() = default;
  template <typename... T>
  float measure(void (GPUbenchmark::*)(T...), const char*, T&&... args);

  // Main interface
  void generalInit(const int deviceId); // Allocate scratch buffers and compute runtime parameters
  void run();                           // Execute all specified callbacks
  void generalFinalize();               // Cleanup
  void printDevices();                  // Dump info

  // Initializations/Finalizations of tests. Not to be measured, in principle used for report
  void readingInit();
  void readingFinalize();

  // Benchmark kernel callbacks
  void readingBenchmark();

 private:
  gpuState<buffer_type> mState;
};

} // namespace benchmark
} // namespace o2
#endif