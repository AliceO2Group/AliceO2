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
/// \author Matteo Concas, Maksym KIzitskyi

#define BOOST_TEST_MODULE Test GPUSMatrixImplementation
#ifdef __HIPCC__
#define GPUPLATFORM "HIP"
#include "hip/hip_runtime.h"
#else
#define GPUPLATFORM "CUDA"
#include <cuda.h>
#endif

#include <iostream>
#include <boost/test/unit_test.hpp>
#include <MathUtils/SMatrixGPU.h>
#include <Math/SMatrix.h>
#include <random>

using MatSym3DGPU = o2::math_utils::detail::SMatrixGPU<double, 3, 3, o2::math_utils::detail::MatRepSymGPU<double, 3>>;
using MatSym3D = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>>;
using Mat3DGPU = o2::math_utils::detail::SMatrixGPU<double, 3, 3, o2::math_utils::detail::MatRepStdGPU<double, 3, 3>>;
using Mat3D = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepStd<double, 3, 3>>;

static constexpr double tolerance = 1e-8;

#define GPU_CHECK(call)                                                                      \
  do {                                                                                       \
    cudaError_t error = call;                                                                \
    if (error != cudaSuccess) {                                                              \
      fprintf(stderr, "CUDA Error: %s (error code %d)\n", cudaGetErrorString(error), error); \
      return;                                                                                \
    }                                                                                        \
  } while (0)

namespace gpu
{
enum PrintMode {
  Decimal,
  Binary,
  Hexadecimal
};

__device__ void floatToBinaryString(float number, char* buffer)
{
  unsigned char* bytePointer = reinterpret_cast<unsigned char*>(&number);
  for (int byteIndex = 3; byteIndex >= 0; --byteIndex) {
    unsigned char byte = bytePointer[byteIndex];
    for (int bitIndex = 7; bitIndex >= 0; --bitIndex) {
      buffer[(3 - byteIndex) * 8 + (7 - bitIndex)] = (byte & (1 << bitIndex)) ? '1' : '0';
    }
  }
  buffer[32] = '\0'; // Null terminator
}

template <typename MatrixType>
GPUd() void printMatrix(const MatrixType& matrix, const char* name, const PrintMode mode)
{
  if (mode == PrintMode::Binary) {
    char buffer[33];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        floatToBinaryString(matrix(i, j), buffer);
        printf("%s(%d,%d) = %s\n", name, i, j, buffer);
      }
    }
  }
  if (mode == PrintMode::Decimal) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        printf("%s(%i,%i) = %f\n", name, i, j, matrix(i, j));
      }
    }
  }
  if (mode == PrintMode::Hexadecimal) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        printf("%s(%d,%d) = %x\n", name, i, j, o2::gpu::CAMath::Float2UIntReint(matrix(i, j)));
      }
    }
  }
}

// Invert test for a single square matrix
template <typename T>
GPUg() void invertMatrixKernelSingle(T* matrix)
{
  matrix->Invert();
}

// Copy test for a single square matrix
template <typename T>
GPUg() void copyMatrixKernelSingle(
  T* srcMatrix,
  T* dstMatrix)
{
  *dstMatrix = *srcMatrix;
}

// Invert test for an array of square matrices
template <typename T>
GPUg() void invertMatrixKernelArray(T* matrices,
                                    const int numMatrices)
{
  for (auto iMatrix = blockIdx.x * blockDim.x + threadIdx.x; iMatrix < numMatrices; iMatrix += blockDim.x * gridDim.x) {
    matrices[iMatrix].Invert();
  }
}

// Copy test for an array of square matrices
template <typename T>
GPUg() void copyMatrixKernelArray(
  T* srcMatrices,
  T* dstMatrices,
  const int numMatrices)
{
  for (auto iMatrix = blockIdx.x * blockDim.x + threadIdx.x; iMatrix < numMatrices; iMatrix += blockDim.x * gridDim.x) {
    srcMatrices[iMatrix] = dstMatrices[iMatrix];
  }
}
} // namespace gpu

// Function to compare two matrices element-wise with a specified tolerance
template <typename MatrixType>
void compareMatricesElementWise(const MatrixType& mat1, const MatrixType& mat2, double tolerance)
{
  auto tol = boost::test_tools::tolerance(tolerance);

  for (unsigned int i = 0; i < mat1.kRows; ++i) {
    for (unsigned int j = 0; j < mat1.kCols; ++j) {
      BOOST_TEST(mat1(i, j) == mat2(i, j), tol);
    }
  }
}

// RAII class for CUDA resources
class GPUMemory
{
 public:
  GPUMemory(size_t size)
  {
    GPU_CHECK(cudaMalloc(&device_ptr, size));
  }
  ~GPUMemory()
  {
    GPU_CHECK(cudaFree(device_ptr));
  }
  void* get() const { return device_ptr; }

 private:
  void* device_ptr;
};

class GPUBenchmark
{
 public:
  GPUBenchmark(const std::string& testName = "") : title(testName)
  {
    GPU_CHECK(cudaEventCreate(&startEvent));
    GPU_CHECK(cudaEventCreate(&stopEvent));
  }
  ~GPUBenchmark()
  {
    GPU_CHECK(cudaEventDestroy(startEvent));
    GPU_CHECK(cudaEventDestroy(stopEvent));
  }
  void start()
  {
    GPU_CHECK(cudaEventRecord(startEvent));
  }
  void stop()
  {
    GPU_CHECK(cudaEventRecord(stopEvent));
    GPU_CHECK(cudaEventSynchronize(stopEvent));
    GPU_CHECK(cudaEventElapsedTime(&duration, startEvent, stopEvent));
  }
  void setTitle(const std::string& newTitle) { title = newTitle; }
  float getDuration() const { return duration; }
  void printDuration() const
  {
    std::cout << "\t - " << title << " kernel execution time: " << duration << " ms" << std::endl;
  }

 private:
  std::string title = "";
  cudaEvent_t startEvent, stopEvent;
  float duration;
};

template <typename T>
void discardResult(const T&)
{
}

void prologue()
{
  int deviceCount;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  if (error != cudaSuccess || !deviceCount) {
    std::cerr << "No " << GPUPLATFORM << " devices found" << std::endl;
    return;
  }

  for (int iDevice = 0; iDevice < deviceCount; ++iDevice) {
    cudaDeviceProp deviceProp;
    discardResult(cudaGetDeviceProperties(&deviceProp, iDevice));
    printf("Testing on: %s, Device %d: %s\n", GPUPLATFORM, iDevice, deviceProp.name);
  }
}

struct GPUSMatrixImplFixtureSolo {
  GPUSMatrixImplFixtureSolo() : SMatrixSym_d(sizeof(MatSym3DGPU)), SMatrixSym_h(), SMatrix_d(sizeof(Mat3DGPU)), SMatrix_h()
  {
    prologue();
    initializeMatrices();
  }

  ~GPUSMatrixImplFixtureSolo() = default;
  void initializeMatrices()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 10.0);

    // Initialize host matrices with random values
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        SMatrixSym_h(i, j) = dis(gen);
        SMatrix_h(i, j) = dis(gen);
      }
    }
    SMatrixSym_original_h = SMatrixSym_h;
    SMatrix_original_h = SMatrix_h;

    // Copy host matrices to device
    GPU_CHECK(cudaMemcpy(SMatrixSym_d.get(), &SMatrixSym_h, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(SMatrix_d.get(), &SMatrix_h, sizeof(Mat3DGPU), cudaMemcpyHostToDevice));
  }

  GPUMemory SMatrixSym_d;
  MatSym3D SMatrixSym_h;
  MatSym3D SMatrixSym_original_h;
  GPUMemory SMatrix_d;
  Mat3D SMatrix_h;
  Mat3D SMatrix_original_h;
};

BOOST_FIXTURE_TEST_CASE(MatrixInversion, GPUSMatrixImplFixtureSolo)
{
  const int nBlocks{1}, nThreads{1};
  GPUBenchmark benchmark("Single symmetric matrix inversion (" + std::to_string(nBlocks) + " blocks, " + std::to_string(nThreads) + " threads)");
  benchmark.start();
  gpu::invertMatrixKernelSingle<MatSym3DGPU><<<nBlocks, nThreads>>>(static_cast<MatSym3DGPU*>(SMatrixSym_d.get()));
  benchmark.stop();
  benchmark.printDuration();
  discardResult(cudaDeviceSynchronize());
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaMemcpy(&SMatrixSym_h, SMatrixSym_d.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  MatSym3D identitySym;
  identitySym(0, 0) = 1;
  identitySym(1, 1) = 1;
  identitySym(2, 2) = 1;
  auto operationSym = SMatrixSym_h * SMatrixSym_original_h;
  MatSym3D resultSym;
  ROOT::Math::AssignSym::Evaluate(resultSym, operationSym);
  compareMatricesElementWise(resultSym, identitySym, tolerance);

  benchmark.setTitle("Single general matrix inversion (" + std::to_string(nBlocks) + " blocks, " + std::to_string(nThreads) + " threads)");
  benchmark.start();
  gpu::invertMatrixKernelSingle<Mat3DGPU><<<nBlocks, nThreads>>>(static_cast<Mat3DGPU*>(SMatrix_d.get()));
  benchmark.stop();
  benchmark.printDuration();
  discardResult(cudaDeviceSynchronize());
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaMemcpy(&SMatrix_h, SMatrix_d.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  Mat3D identity;
  identity(0, 0) = 1;
  identity(1, 1) = 1;
  identity(2, 2) = 1;
  auto operation = SMatrix_h * SMatrix_original_h;
  Mat3D result;
  ROOT::Math::Assign<double, 3, 3, decltype(operation), ROOT::Math::MatRepStd<double, 3, 3>, ROOT::Math::MatRepStd<double, 3, 3>>::Evaluate(result, operation);
  compareMatricesElementWise(result, identity, tolerance);
}

struct GPUSMatrixImplFixtureDuo {
  GPUSMatrixImplFixtureDuo() : SMatrixSym_d_A(sizeof(MatSym3DGPU)), SMatrixSym_h_A(), SMatrix_d_A(sizeof(Mat3DGPU)), SMatrix_h_A(), SMatrixSym_d_B(sizeof(MatSym3DGPU)), SMatrixSym_h_B(), SMatrix_d_B(sizeof(Mat3DGPU)), SMatrix_h_B()
  {
    prologue();
    initializeMatrices();
  }

  ~GPUSMatrixImplFixtureDuo() = default;

  void initializeMatrices()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 10.0);

    // Initialize host matrices with random values
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        SMatrixSym_h_A(i, j) = dis(gen);
        SMatrix_h_A(i, j) = dis(gen);

        SMatrixSym_h_B(i, j) = dis(gen);
        SMatrix_h_B(i, j) = dis(gen);
      }
    }

    // Copy host matrices to device
    GPU_CHECK(cudaMemcpy(SMatrixSym_d_A.get(), &SMatrixSym_h_A, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(SMatrix_d_A.get(), &SMatrix_h_A, sizeof(Mat3DGPU), cudaMemcpyHostToDevice));

    GPU_CHECK(cudaMemcpy(SMatrixSym_d_B.get(), &SMatrixSym_h_B, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(SMatrix_d_B.get(), &SMatrix_h_B, sizeof(Mat3DGPU), cudaMemcpyHostToDevice));
  }

  GPUMemory SMatrixSym_d_A;
  MatSym3D SMatrixSym_h_A;

  GPUMemory SMatrixSym_d_B;
  MatSym3D SMatrixSym_h_B;

  GPUMemory SMatrix_d_A;
  Mat3D SMatrix_h_A;

  GPUMemory SMatrix_d_B;
  Mat3D SMatrix_h_B;
};

BOOST_FIXTURE_TEST_CASE(TestMatrixCopyingAndComparison, GPUSMatrixImplFixtureDuo)
{
  const int nBlocks{1}, nThreads{1};
  GPUBenchmark benchmark("Single symmetric matrix copy (" + std::to_string(nBlocks) + " blocks, " + std::to_string(nThreads) + " threads)");
  benchmark.start();
  gpu::copyMatrixKernelSingle<MatSym3DGPU><<<nBlocks, nThreads>>>(static_cast<MatSym3DGPU*>(SMatrixSym_d_A.get()), static_cast<MatSym3DGPU*>(SMatrixSym_d_B.get()));
  benchmark.stop();
  benchmark.printDuration();
  discardResult(cudaDeviceSynchronize());
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaMemcpy(&SMatrixSym_h_B, SMatrixSym_d_B.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  compareMatricesElementWise(SMatrixSym_h_A, SMatrixSym_h_B, 0.0);

  benchmark.setTitle("Single general matrix copy (" + std::to_string(nBlocks) + " blocks, " + std::to_string(nThreads) + " threads)");
  benchmark.start();
  gpu::copyMatrixKernelSingle<Mat3DGPU><<<nBlocks, nThreads>>>(static_cast<Mat3DGPU*>(SMatrix_d_A.get()), static_cast<Mat3DGPU*>(SMatrix_d_B.get()));
  benchmark.stop();
  benchmark.printDuration();
  discardResult(cudaDeviceSynchronize());
  GPU_CHECK(cudaGetLastError());

  GPU_CHECK(cudaMemcpy(&SMatrix_h_B, SMatrix_d_B.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  compareMatricesElementWise(SMatrix_h_A, SMatrix_h_B, 0.0);
}
template <size_t D = 1000>
struct GPUSmatrixImplFixtureSoloArray {
  GPUSmatrixImplFixtureSoloArray() : SMatrixSymArray_d(D * sizeof(MatSym3DGPU)), SMatrixArray_d(D * sizeof(Mat3DGPU))
  {
    SMatrixSymVector_h.resize(D);
    SMatrixVector_h.resize(D);
    SMatrixSym_original_h.resize(D);
    SMatrix_original_h.resize(D);
    prologue();
    initializeMatrices();
  }

  ~GPUSmatrixImplFixtureSoloArray() = default;
  void initializeMatrices()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 10.0);

    // Initialize host matrices with random values
    for (size_t iMatrix{0}; iMatrix < D; ++iMatrix) {
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          SMatrixSymVector_h[iMatrix](i, j) = dis(gen);
          SMatrixVector_h[iMatrix](i, j) = dis(gen);
        }
      }

      SMatrixSym_original_h[iMatrix] = SMatrixSymVector_h[iMatrix];
      SMatrix_original_h[iMatrix] = SMatrixVector_h[iMatrix];
    }

    // Copy host matrices to device
    GPU_CHECK(cudaMemcpy(SMatrixSymArray_d.get(), SMatrixSymVector_h.data(), D * sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(SMatrixArray_d.get(), SMatrixVector_h.data(), D * sizeof(Mat3DGPU), cudaMemcpyHostToDevice));
  }

  GPUMemory SMatrixSymArray_d;
  std::vector<MatSym3D> SMatrixSymVector_h;
  std::vector<MatSym3D> SMatrixSym_original_h;
  GPUMemory SMatrixArray_d;
  std::vector<Mat3D> SMatrixVector_h;
  std::vector<Mat3D> SMatrix_original_h;
};

BOOST_FIXTURE_TEST_CASE(MatrixInversionArray, GPUSmatrixImplFixtureSoloArray<1'000'000>)
{
  const int nBlocks{20}, nThreads{512};
  GPUBenchmark benchmark("Array of 1'000'000 symmetric matrices inversion (" + std::to_string(nBlocks) + " blocks, " + std::to_string(nThreads) + " threads)");
  benchmark.start();
  gpu::invertMatrixKernelArray<MatSym3DGPU><<<nBlocks, nThreads>>>(static_cast<MatSym3DGPU*>(SMatrixSymArray_d.get()), 1'000'000);
  benchmark.stop();
  benchmark.printDuration();
  discardResult(cudaDeviceSynchronize());
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaMemcpy(SMatrixSymVector_h.data(), SMatrixSymArray_d.get(), 1'000'000 * sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  for (size_t iMatrix{0}; iMatrix < 1'000'000; ++iMatrix) {
    // Cross-check with the CPU implementation
    SMatrixSym_original_h[iMatrix].Invert();
    compareMatricesElementWise(SMatrixSymVector_h[iMatrix], SMatrixSym_original_h[iMatrix], tolerance);
  }

  benchmark.setTitle("Array of 1'000'000 general matrices inversion (" + std::to_string(nBlocks) + " blocks, " + std::to_string(nThreads) + " threads)");
  benchmark.start();
  gpu::invertMatrixKernelArray<Mat3DGPU><<<nBlocks, nThreads>>>(static_cast<Mat3DGPU*>(SMatrixArray_d.get()), 1'000'000);
  benchmark.stop();
  benchmark.printDuration();
  discardResult(cudaDeviceSynchronize());
  GPU_CHECK(cudaGetLastError());
  GPU_CHECK(cudaMemcpy(SMatrixVector_h.data(), SMatrixArray_d.get(), 1'000'000 * sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  for (size_t iMatrix{0}; iMatrix < 1'000'000; ++iMatrix) {
    // Cross-check with the CPU implementation
    SMatrix_original_h[iMatrix].Invert();
    compareMatricesElementWise(SMatrixVector_h[iMatrix], SMatrix_original_h[iMatrix], tolerance);
  }
}