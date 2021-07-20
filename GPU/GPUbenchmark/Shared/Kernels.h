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
/// \file Kernels.h
/// \author: mconcas@cern.ch

#ifndef GPU_BENCHMARK_KERNELS_H
#define GPU_BENCHMARK_KERNELS_H

#include "Utils.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>

namespace o2
{
namespace benchmark
{

template <class chunk_type>
class GPUbenchmark final
{
 public:
  GPUbenchmark() = delete; // need for a configuration
  GPUbenchmark(benchmarkOpts& opts, std::shared_ptr<ResultWriter> rWriter) : mResultWriter{rWriter}, mOptions{opts}
  {
  }
  virtual ~GPUbenchmark() = default;
  template <typename... T>
  float measure(void (GPUbenchmark::*)(T...), const char*, T&&... args);

  // Single stream synchronous (sequential kernels) execution
  template <typename... T>
  float benchmarkSync(void (*kernel)(T...),
                      int nLaunches, int blocks, int threads, T&... args);

  // Multi-streams asynchronous executions on whole memory
  template <typename... T>
  std::vector<float> benchmarkAsync(void (*kernel)(int, T...),
                                    int nStreams, int nLaunches, int blocks, int threads, T&... args);

  // Main interface
  void globalInit();     // Allocate scratch buffers and compute runtime parameters
  void run();            // Execute all specified callbacks
  void globalFinalize(); // Cleanup
  void printDevices();   // Dump info

  // Initializations/Finalizations of tests. Not to be measured, in principle used for report
  void readInit();
  void readFinalize();

  void writeInit();
  void writeFinalize();

  void copyInit();
  void copyFinalize();

  // Kernel calling wrappers
  void readSequential(SplitLevel sl);
  void readConcurrent(SplitLevel sl, int nRegions = 2);

  void writeSequential(SplitLevel sl);
  void writeConcurrent(SplitLevel sl, int nRegions = 2);

  void copySequential(SplitLevel sl);
  void copyConcurrent(SplitLevel sl, int nRegions = 2);

 private:
  gpuState<chunk_type> mState;
  std::shared_ptr<ResultWriter> mResultWriter;
  benchmarkOpts mOptions;
};

} // namespace benchmark
} // namespace o2
#endif