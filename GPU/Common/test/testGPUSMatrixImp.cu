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

/// \file testGPUSMatrixImp.cu
/// \author Matteo Concas

#define BOOST_TEST_MODULE Test GPUSMatrixImpl
#ifdef __HIPCC__
#define GPUPLATFORM "HIP"
#include "hip/hip_runtime.h"
#else
#define GPUPLATFORM "CUDA"
#endif

#include <boost/test/unit_test.hpp>
#include <iostream>

cudaError_t gpuCheckError(cudaError_t gpuErrorCode)
{
  if (gpuErrorCode != cudaSuccess) {
    std::cerr << "ErrorCode " << gpuErrorCode << " " << cudaGetErrorName(gpuErrorCode) << ": " << cudaGetErrorString(gpuErrorCode) << std::endl;
    exit(-1);
  }
  return gpuErrorCode;
}

__global__ void kernel()
{
  printf("Hello world from device\n");
}

struct GPUSMatrixImplFixture {
  GPUSMatrixImplFixture()
  {
    std::cout << "GPUSMatrixImplFixture" << std::endl;
    // Get the number of GPU devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
      std::cerr << "No " << GPUPLATFORM << " devices found" << std::endl;
    }

    for (int iDevice = 0; iDevice < deviceCount; ++iDevice) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, iDevice);

      std::cout << GPUPLATFORM << " Device " << iDevice << ": " << deviceProp.name << std::endl;
    }

    kernel<<<1, 1>>>();
    gpuCheckError(cudaDeviceSynchronize());
    i = 3;
  }

  ~GPUSMatrixImplFixture()
  {
    std::cout << "~GPUSMatrixImplFixture" << std::endl;
  }

  int i;
};

BOOST_FIXTURE_TEST_CASE(DummyFixtureUsage, GPUSMatrixImplFixture)
{
  BOOST_TEST(i == 3);
}