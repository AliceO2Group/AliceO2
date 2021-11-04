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

#if defined(__HIPCC__)
#include "hip/hip_runtime.h"
#endif

#include <iostream>
#include <sstream>
#include <iomanip>
#include <typeinfo>
#include <boost/program_options.hpp>
#include <vector>
#include <string>
#include <TTree.h>
#include <TFile.h>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define configLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define GB (1024 * 1024 * 1024)

#define failed(...)                       \
  printf("%serror: ", KRED);              \
  printf(__VA_ARGS__);                    \
  printf("\n");                           \
  printf("error: TEST FAILED\n%s", KNRM); \
  exit(EXIT_FAILURE);
#endif

enum class Test {
  Read,
  Write,
  Copy
};

inline std::ostream& operator<<(std::ostream& os, Test test)
{
  switch (test) {
    case Test::Read:
      os << "read";
      break;
    case Test::Write:
      os << "write";
      break;
    case Test::Copy:
      os << "copy";
      break;
  }
  return os;
}

enum class Mode {
  Sequential,
  Concurrent,
  Distributed
};

inline std::ostream& operator<<(std::ostream& os, Mode mode)
{
  switch (mode) {
    case Mode::Sequential:
      os << "sequential";
      break;
    case Mode::Concurrent:
      os << "concurrent";
      break;
    case Mode::Distributed:
      os << "distributed";
      break;
  }
  return os;
}

enum class KernelConfig {
  Single,
  Multi,
  All
};

inline std::ostream& operator<<(std::ostream& os, KernelConfig config)
{
  switch (config) {
    case KernelConfig::Single:
      os << "single";
      break;
    case KernelConfig::Multi:
      os << "multiple";
      break;
    case KernelConfig::All:
      os << "all";
      break;
  }
  return os;
}

template <class T>
inline std::string getType()
{
  if (typeid(T).name() == typeid(char).name()) {
    return std::string{"char"};
  }
  if (typeid(T).name() == typeid(size_t).name()) {
    return std::string{"unsigned_long"};
  }
  if (typeid(T).name() == typeid(int).name()) {
    return std::string{"int"};
  }
  if (typeid(T).name() == typeid(int4).name()) {
    return std::string{"int4"};
  }
  return std::string{"unknown"};
}

inline std::string getTestName(Mode mode, Test test, KernelConfig blocks)
{
  std::string tname;
  tname += (mode == Mode::Sequential) ? "seq_" : "conc_";
  tname += (test == Test::Read) ? "read_" : (test == Test::Write) ? "write_"
                                                                  : "copy_";
  tname += (blocks == KernelConfig::Single) ? "SB" : "MB";
  return tname;
}

// Return pointer to custom offset (GB)
template <class chunk_t>
inline chunk_t* getCustomPtr(chunk_t* scratchPtr, float startGB)
{
  return reinterpret_cast<chunk_t*>(reinterpret_cast<char*>(scratchPtr) + static_cast<size_t>(GB * startGB));
}

inline float computeThroughput(Test test, float result, float chunkSizeGB, int ntests)
{
  // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
  // Eff_bandwidth (GB/s) = (B_r + B_w) / (~1e9 * Time (s))

  return 1e3 * chunkSizeGB * ntests / result;
}

template <class chunk_t>
inline size_t getBufferCapacity(float chunkReservedGB)
{
  return static_cast<size_t>((GB * chunkReservedGB) / sizeof(chunk_t));
}

// LCG: https://rosettacode.org/wiki/Linear_congruential_generator
class LCGRnd
{
 public:
  __host__ __device__ void seed(unsigned int s) { mSeed = s; }

 protected:
  __host__ __device__ LCGRnd() : mSeed{0}, mA{0}, mC{0}, mM(2147483648) {}
  __host__ __device__ int rnd() { return (mSeed = (mA * mSeed + mC) % mM); }

  int mA, mC;
  unsigned int mM, mSeed;
};

class BSDRnd : public LCGRnd
{
 public:
  __host__ __device__ BSDRnd()
  {
    mA = 1103515245;
    mC = 12345;
  }
  __host__ __device__ int rnd() { return LCGRnd::rnd(); }
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
  std::vector<KernelConfig> pools = {KernelConfig::Single, KernelConfig::Multi};
  std::vector<std::string> dtypes = {"char", "int", "ulong"};
  std::vector<std::pair<float, float>> testChunks;
  float chunkReservedGB = 1.f;
  float threadPoolFraction = 1.f;
  float freeMemoryFractionToAllocate = 0.95f;
  int kernelLaunches = 1;
  int nTests = 1;
  int streams = 8;
  std::string outFileName = "benchmark_result";
};

template <class chunk_t>
struct gpuState {
  int getMaxChunks()
  {
    return (double)scratchSize / (chunkReservedGB * GB);
  }

  size_t getChunkCapacity()
  {
    return getBufferCapacity<chunk_t>(chunkReservedGB);
  }

  int getNKernelLaunches() { return iterations; }
  int getStreamsPoolSize() { return streams; }

  // Configuration
  size_t nMaxThreadsPerDimension;
  int iterations;
  int streams;

  float chunkReservedGB; // Size of each partition (GB)

  // General containers and state
  chunk_t* scratchPtr;                             // Pointer to scratch buffer
  size_t scratchSize;                              // Size of scratch area (B)
  std::vector<chunk_t*> partAddrOnHost;            // Pointers to scratch partitions on host vector
  std::vector<std::pair<float, float>> testChunks; // Vector of definitions for arbitrary chunks

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
  void storeBenchmarkEntry(Test test, int chunk, float entry, float chunkSizeGB, int nLaunches);
  void addBenchmarkEntry(const std::string bName, const std::string type, const int nChunks);
  void snapshotBenchmark();
  void saveToFile();

 private:
  std::vector<float> mTimeResults;
  std::vector<TTree*> mTimeTrees;
  std::vector<float> mThroughputResults;
  std::vector<TTree*> mThroughputTrees;
  TFile* mOutfile;
};

inline ResultWriter::ResultWriter(const std::string resultsTreeFilename)
{
  mOutfile = TFile::Open(resultsTreeFilename.data(), "recreate");
}

inline void ResultWriter::addBenchmarkEntry(const std::string bName, const std::string type, const int nChunks)
{
  mTimeTrees.emplace_back(new TTree((bName + "_" + type).data(), (bName + "_" + type).data()));
  mTimeResults.clear();
  mTimeResults.resize(nChunks);
  mTimeTrees.back()->Branch("elapsed", &mTimeResults);

  mThroughputTrees.emplace_back(new TTree((bName + "_" + type + "_TP").data(), (bName + "_" + type + "_TP").data()));
  mThroughputResults.clear();
  mThroughputResults.resize(nChunks);
  mThroughputTrees.back()->Branch("throughput", &mThroughputResults);
}

inline void ResultWriter::storeBenchmarkEntry(Test test, int chunk, float entry, float chunkSizeGB, int nLaunches)
{
  mTimeResults[chunk] = entry;
  mThroughputResults[chunk] = computeThroughput(test, entry, chunkSizeGB, nLaunches);
}

inline void ResultWriter::snapshotBenchmark()
{
  mTimeTrees.back()->Fill();
  mThroughputTrees.back()->Fill();
}

inline void ResultWriter::saveToFile()
{
  mOutfile->cd();
  for (auto t : mTimeTrees) {
    t->Write();
  }
  for (auto t : mThroughputTrees) {
    t->Write();
  }
  mOutfile->Close();
}

} // namespace benchmark
} // namespace o2