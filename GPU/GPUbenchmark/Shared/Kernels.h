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

#include "GPUCommonDef.h"
#include "Common.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

// #define PARTITION_SIZE_GB 1
// #define FREE_MEMORY_FRACTION_TO_ALLOCATE 0.95f

namespace o2
{
namespace benchmark
{

template <class buffer_type>
class GPUbenchmark final
{
 public:
  GPUbenchmark() = delete; // need for a configuration
  GPUbenchmark(benchmarkOpts& opts) : mOptions{opts}
  {
  }
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
  void readingBenchmark(size_t iterations);

 private:
  gpuState<buffer_type> mState;
  benchmarkOpts mOptions;
};

} // namespace benchmark
} // namespace o2
#endif