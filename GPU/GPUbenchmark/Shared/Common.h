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
///
/// \file Common.h
/// \author: mconcas@cern.ch

#ifndef GPU_BENCHMARK_COMMON_H
#define GPU_BENCHMARK_COMMON_H

#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <boost/program_options.hpp>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define GB (1024 * 1024 * 1024)

namespace o2
{
namespace benchmark
{
struct benchmarkOpts {
  benchmarkOpts() = default;

  float partitionSizeGB = 1.f;
  float freeMemoryFractionToAllocate = 0.95f;
  size_t iterations = 1;
};

template <class T>
struct gpuState {
  int getMaxSegments()
  {
    return (double)scratchSize / (partitionSizeGB * GB);
  }

  void computeScratchPtrs()
  {
    partAddrOnHost.resize(getMaxSegments());
    for (size_t iBuffAddress{0}; iBuffAddress < getMaxSegments(); ++iBuffAddress) {
      partAddrOnHost[iBuffAddress] = reinterpret_cast<T*>(reinterpret_cast<char*>(scratchPtr) + static_cast<size_t>(GB * partitionSizeGB) * iBuffAddress);
    }
  }

  size_t getPartitionCapacity()
  {
    return static_cast<size_t>(GB * partitionSizeGB / sizeof(T));
  }

  std::vector<T*> getScratchPtrs()
  {
    return partAddrOnHost;
  }

  std::vector<std::vector<T>>& getHostBuffers()
  {
    return gpuBuffersHost;
  }

  size_t getNiterations() { return iterations; }

  // Configuration
  size_t nMaxThreadsPerDimension;
  size_t iterations;

  float partitionSizeGB; // Size of each partition (GB)

  // General containers and state
  T* scratchPtr;                              // Pointer to scratch buffer
  size_t scratchSize;                         // Size of scratch area (B)
  std::vector<T*> partAddrOnHost;             // Pointers to scratch partitions on host vector
  std::vector<std::vector<T>> gpuBuffersHost; // Host-based vector-ized data

  // Test-specific containers
  T* deviceReadingResultsPtr;              // Results of the reading test (single variable) on GPU
  std::vector<T> hostReadingResultsVector; // Results of the reading test (single variable) on host

  // Static info
  size_t totalMemory;
  size_t nMultiprocessors;
  size_t nMaxThreadsPerBlock;
};

} // namespace benchmark
} // namespace o2

#define failed(...)                       \
  printf("%serror: ", KRED);              \
  printf(__VA_ARGS__);                    \
  printf("\n");                           \
  printf("error: TEST FAILED\n%s", KNRM); \
  exit(EXIT_FAILURE);

#endif