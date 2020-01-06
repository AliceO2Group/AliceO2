// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testGPUsortCUDA.cu
/// \author ...

#define GPUCA_GPUTYPE_PASCAL

#define BOOST_TEST_MODULE Test GPUCommonAlgorithm Sorting CUDA
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>
#include <cstring>
#include <boost/test/unit_test.hpp>
#include "GPUCommonAlgorithm.h"

///////////////////////////////////////////////////////////////
// Test setup and tear down
///////////////////////////////////////////////////////////////

static constexpr float TOLERANCE = 10 * std::numeric_limits<float>::epsilon();

cudaError_t cudaCheckError(cudaError_t cudaErrorCode)
{
  if (cudaErrorCode != cudaSuccess) {
    std::cerr << "ErrorCode " << cudaErrorCode << " " << cudaGetErrorName(cudaErrorCode) << ": " << cudaGetErrorString(cudaErrorCode) << std::endl;
    exit(-1);
  }
  return cudaErrorCode;
}

struct TestEnvironment {
  TestEnvironment() : size(101), data(nullptr), sorted(size)
  {
    cudaCheckError(cudaMallocManaged(&data, size * sizeof(float)));

    // create an array of unordered floats with negative and positive values
    for (size_t i = 0; i < size; i++) {
      data[i] = size / 2 - i;
    }
    // create copy
    std::memcpy(sorted.data(), data, size * sizeof(float));
    // sort
    std::sort(sorted.begin(), sorted.end());
  }

  ~TestEnvironment()
  {
    cudaFree(data);
  };

  const size_t size;
  float* data;
  std::vector<float> sorted;
};

template <typename T>
void testAlmostEqualArray(T* correct, T* testing, size_t size)
{
  for (size_t i = 0; i < size; i++) {
    if (std::fabs(correct[i]) < TOLERANCE) {
      BOOST_CHECK_SMALL(testing[i], TOLERANCE);
    } else {
      BOOST_CHECK_CLOSE(correct[i], testing[i], 1.0 / TOLERANCE);
    }
  }
}

///////////////////////////////////////////////////////////////

__global__ void sortInThread(float* data, size_t dataLength)
{
  // make sure only one thread is working on this.
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    o2::gpu::CAAlgo::sort(data, data + dataLength);
  }
}

__global__ void sortInThreadWithOperator(float* data, size_t dataLength)
{
  // make sure only one thread is working on this.
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    o2::gpu::CAAlgo::sort(data, data + dataLength, [](float a, float b) { return a < b; });
  }
}

///////////////////////////////////////////////////////////////

__global__ void sortInBlock(float* data, size_t dataLength)
{
  o2::gpu::CAAlgo::sortInBlock<float>(data, data + dataLength);
}

__global__ void sortInBlockWithOperator(float* data, size_t dataLength)
{
  o2::gpu::CAAlgo::sortInBlock(data, data + dataLength, [](float a, float b) { return a < b; });
}
///////////////////////////////////////////////////////////////

BOOST_AUTO_TEST_SUITE(TestsortInThread)

BOOST_FIXTURE_TEST_CASE(GPUsortThreadCUDA, TestEnvironment)
{
  sortInThread<<<1, 1>>>(data, size);
  BOOST_CHECK_EQUAL(cudaCheckError(cudaDeviceSynchronize()), CUDA_SUCCESS);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_FIXTURE_TEST_CASE(GPUsortThreadOperatorCUDA, TestEnvironment)
{
  sortInThreadWithOperator<<<1, 1>>>(data, size);
  BOOST_CHECK_EQUAL(cudaCheckError(cudaDeviceSynchronize()), CUDA_SUCCESS);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TestsortInBlock)

BOOST_FIXTURE_TEST_CASE(GPUsortBlockCUDA, TestEnvironment)
{
  sortInBlock<<<1, 128>>>(data, size);
  BOOST_CHECK_EQUAL(cudaCheckError(cudaDeviceSynchronize()), CUDA_SUCCESS);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_FIXTURE_TEST_CASE(GPUsortBlockOperatorCUDA, TestEnvironment)
{
  sortInBlockWithOperator<<<1, 128>>>(data, size);
  BOOST_CHECK_EQUAL(cudaCheckError(cudaDeviceSynchronize()), CUDA_SUCCESS);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_AUTO_TEST_SUITE_END()
