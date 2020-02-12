// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testGPUsortHIP.hip.cxx
/// \author Michael Lettrich

#define GPUCA_GPUTYPE_VEGA

#define BOOST_TEST_MODULE Test GPUCommonAlgorithm Sorting HIP
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>
#include <cstring>
#include <hip/hip_runtime.h>
#include <boost/test/unit_test.hpp>
#include "GPUCommonAlgorithm.h"

///////////////////////////////////////////////////////////////
// Test setup and tear down
///////////////////////////////////////////////////////////////

static constexpr float TOLERANCE = 10 * std::numeric_limits<float>::epsilon();

hipError_t hipCheckError(hipError_t hipErrorCode)
{
  if (hipErrorCode != hipSuccess) {
    std::cerr << "ErrorCode " << hipErrorCode << " " << hipGetErrorName(hipErrorCode) << ": " << hipGetErrorString(hipErrorCode) << std::endl;
  }
  return hipErrorCode;
}

void hipCheckErrorFatal(hipError_t hipErrorCode)
{
  if (hipCheckError(hipErrorCode) != hipSuccess) {
    exit(-1);
  }
}

struct TestEnvironment {
  TestEnvironment() : size(101), data(nullptr), sorted(size)
  {
    hipCheckErrorFatal(hipHostMalloc(&data, size * sizeof(float), hipHostRegisterDefault));

    // create an array of unordered floats with negative and positive values
    for (size_t i = 0; i < size; i++) {
      data[i] = size / 2.0 - i;
    }
    // create copy
    std::memcpy(sorted.data(), data, size * sizeof(float));
    // sort
    std::sort(sorted.begin(), sorted.end());
  }

  ~TestEnvironment() // NOLINT: clang-tidy doesn't understand hip macro magic, and thinks this is trivial
  {
    hipCheckErrorFatal(hipFree(data));
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
      BOOST_CHECK_CLOSE(correct[i], testing[i], TOLERANCE);
    }
  }
}

///////////////////////////////////////////////////////////////

__global__ void sortInThread(float* data, size_t dataLength)
{
  // make sure only one thread is working on this.
  if (hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0 && hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0) {
    o2::gpu::CAAlgo::sort(data, data + dataLength);
  }
}

__global__ void sortInThreadWithOperator(float* data, size_t dataLength)
{
  // make sure only one thread is working on this.
  if (hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0 && hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0) {
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

BOOST_FIXTURE_TEST_CASE(GPUsortThreadHIP, TestEnvironment)
{
  hipLaunchKernelGGL(sortInThread, dim3(1), dim3(1), 0, 0, data, size);
  // sortInThread<<<dim3(1), dim3(1), 0, 0>>>(data, size);
  BOOST_CHECK_EQUAL(hipCheckError(hipDeviceSynchronize()), hipSuccess);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_FIXTURE_TEST_CASE(GPUsortThreadOperatorHIP, TestEnvironment)
{
  hipLaunchKernelGGL(sortInThreadWithOperator, dim3(1), dim3(1), 0, 0, data, size);
  // sortInThreadWithOperator<<<dim3(1), dim3(1), 0, 0>>>(data, size);
  BOOST_CHECK_EQUAL(hipCheckError(hipDeviceSynchronize()), hipSuccess);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TestsortInBlock)

BOOST_FIXTURE_TEST_CASE(GPUsortBlockHIP, TestEnvironment)
{
  hipLaunchKernelGGL(sortInBlock, dim3(1), dim3(128), 0, 0, data, size);
  // sortInBlock<<<dim3(1), dim3(128), 0, 0>>>(data, size);
  BOOST_CHECK_EQUAL(hipCheckError(hipDeviceSynchronize()), hipSuccess);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_FIXTURE_TEST_CASE(GPUsortBlockOperatorHIP, TestEnvironment)
{
  hipLaunchKernelGGL(sortInBlockWithOperator, dim3(1), dim3(128), 0, 0, data, size);
  // sortInBlockWithOperator<<<dim3(1), dim3(128), 0, 0>>>(data, size);
  BOOST_CHECK_EQUAL(hipCheckError(hipDeviceSynchronize()), hipSuccess);
  testAlmostEqualArray(sorted.data(), data, size);
}

BOOST_AUTO_TEST_SUITE_END()
