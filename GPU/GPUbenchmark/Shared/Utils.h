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
#include <vector>
#include <TTree.h>
#include <TFile.h>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define GB (1024 * 1024 * 1024)

enum class Test {
  Read,
  Write,
  Copy
};

enum class Mode {
  Sequential,
  Concurrent
};

enum class SplitLevel {
  Blocks,
  Threads
};

namespace o2
{
namespace benchmark
{

struct benchmarkOpts {
  benchmarkOpts() = default;

  int deviceId = 0;
  std::vector<Test> tests = {Test::Read, Test::Write, Test::Copy};
  std::vector<Mode> modes = {Mode::Sequential, Mode::Concurrent};
  std::vector<SplitLevel> pools = {SplitLevel::Blocks, SplitLevel::Threads};
  float chunkReservedGB = 1.f;
  int nRegions = 2;
  float freeMemoryFractionToAllocate = 0.95f;
  int kernelLaunches = 1;
  int nTests = 1;
};

template <class T>
struct gpuState {
  int getMaxChunks()
  {
    return (double)scratchSize / (chunkReservedGB * GB);
  }

  void computeScratchPtrs()
  {
    partAddrOnHost.resize(getMaxChunks());
    for (size_t iBuffAddress{0}; iBuffAddress < getMaxChunks(); ++iBuffAddress) {
      partAddrOnHost[iBuffAddress] = reinterpret_cast<T*>(reinterpret_cast<char*>(scratchPtr) + static_cast<size_t>(GB * chunkReservedGB) * iBuffAddress);
    }
  }

  size_t getPartitionCapacity()
  {
    return static_cast<size_t>(GB * chunkReservedGB / sizeof(T));
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

  float chunkReservedGB; // Size of each partition (GB)

  // General containers and state
  T* scratchPtr;                              // Pointer to scratch buffer
  size_t scratchSize;                         // Size of scratch area (B)
  std::vector<T*> partAddrOnHost;             // Pointers to scratch partitions on host vector
  std::vector<std::vector<T>> gpuBuffersHost; // Host-based vector-ized data
  T* deviceReadResultsPtr;                    // Results of the read test (single variable) on GPU
  std::vector<T> hostReadResultsVector;       // Results of the read test (single variable) on host
  T* deviceWriteResultsPtr;                   // Results of the write test (single variable) on GPU
  std::vector<T> hostWriteResultsVector;      // Results of the write test (single variable) on host
  T* deviceCopyInputsPtr;                     // Inputs of the copy test (single variable) on GPU
  std::vector<T> hostCopyInputsVector;        // Inputs of the copy test (single variable) on host

  // Static info
  size_t totalMemory;
  size_t nMultiprocessors;
  size_t nMaxThreadsPerBlock;
};

// Interface class to stream results to root file
class ResultWriter
{
 public:
  explicit ResultWriter(const std::string resultsTreeFilename = "benchmark_results.root");
  ~ResultWriter() = default;
  void storeBenchmarkEntry(int chunk, float entry);
  void addBenchmarkEntry(const std::string bName, const std::string type, const int nChunks);
  void snapshotBenchmark();
  void saveToFile();

 private:
  std::vector<float> mBenchmarkResults;
  std::vector<TTree*> mBenchmarkTrees;
  TFile* mOutfile;
};

inline ResultWriter::ResultWriter(const std::string resultsTreeFilename)
{
  mOutfile = TFile::Open(resultsTreeFilename.data(), "recreate");
}

inline void ResultWriter::addBenchmarkEntry(const std::string bName, const std::string type, const int nChunks)
{
  mBenchmarkTrees.emplace_back(new TTree((bName + "_" + type).data(), (bName + "_" + type).data()));
  mBenchmarkResults.clear();
  mBenchmarkResults.resize(nChunks);
  mBenchmarkTrees.back()->Branch("elapsed", &mBenchmarkResults);
}

inline void ResultWriter::storeBenchmarkEntry(int chunk, float entry)
{
  mBenchmarkResults[chunk] = entry;
}

inline void ResultWriter::snapshotBenchmark()
{
  mBenchmarkTrees.back()->Fill();
}

inline void ResultWriter::saveToFile()
{
  mOutfile->cd();
  for (auto t : mBenchmarkTrees) {
    t->Write();
  }
  mOutfile->Close();
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