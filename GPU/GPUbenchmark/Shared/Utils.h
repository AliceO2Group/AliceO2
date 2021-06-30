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

#ifndef GPU_BENCHMARK_UTILS_H
#define GPU_BENCHMARK_UTILS_H

#include <iostream>
#include <iomanip>
#include <typeinfo>
#include <boost/program_options.hpp>
#include "CommonUtils/TreeStreamRedirector.h"

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

enum class SplitLevel {
  Blocks,
  Threads
};

struct benchmarkOpts {
  benchmarkOpts() = default;

  float partitionSizeGB = 1.f;
  float freeMemoryFractionToAllocate = 0.95f;
  int kernelLaunches = 1;
  int nTests = 1;
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

  int getNKernelLaunches() { return iterations; }

  // Configuration
  size_t nMaxThreadsPerDimension;
  int iterations;

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

// Interface class to stream results to root file
class ResultStreamer
{
 public:
  explicit ResultStreamer(const std::string debugTreeFileName = "benchmark_results.root");
  ~ResultStreamer();
  void storeBenchmarkEntry(std::string benchmarkName, std::string split, std::string type, float entry);

 private:
  std::string mDebugTreeFileName = "benchmark_results.root"; // output filename
  o2::utils::TreeStreamRedirector* mTreeStream;              // observer
};

inline ResultStreamer::ResultStreamer(const std::string debugTreeFileName)
{
  mDebugTreeFileName = debugTreeFileName;
  mTreeStream = new o2::utils::TreeStreamRedirector(debugTreeFileName.data(), "recreate");
}

inline ResultStreamer::~ResultStreamer()
{
  delete mTreeStream;
}

inline void ResultStreamer::storeBenchmarkEntry(std::string benchmarkName, std::string split, std::string type, float entry)
{
  (*mTreeStream)
    << (benchmarkName + "_" + type + "_" + split).data()
    << "elapsed=" << entry
    << "\n";
}

} // namespace benchmark
} // namespace o2

#define failed(...)                       \
  printf("%serror: ", KRED);              \
  printf(__VA_ARGS__);                    \
  printf("\n");                           \
  printf("error: TEST FAILED\n%s", KNRM); \
  exit(EXIT_FAILURE);

#endif