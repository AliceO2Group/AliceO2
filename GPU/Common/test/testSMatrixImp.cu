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

/// \file testGPUSMatrixImpUnified.cu
/// \author Matteo Concas

#define BOOST_TEST_MODULE Test GPUSMatrixImpl
#ifdef __HIPCC__
#define GPUPLATFORM "HIP"
#include "hip/hip_runtime.h"
#else
#define GPUPLATFORM "CUDA"
#include <cuda.h>
#endif

#include <boost/test/unit_test.hpp>
#include <MathUtils/SMatrixGPU.h>
#include <Math/SMatrix.h>
#include <random>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                                     \
  do {                                                                                       \
    cudaError_t error = call;                                                                \
    if (error != cudaSuccess) {                                                              \
      fprintf(stderr, "CUDA Error: %s (error code %d)\n", cudaGetErrorString(error), error); \
      return;                                                                                \
    }                                                                                        \
  } while (0)

// RAII class for CUDA resources
class CudaMemory
{
 public:
  CudaMemory(size_t size)
  {
    CUDA_CHECK(cudaMalloc(&device_ptr, size));
  }
  ~CudaMemory()
  {
    cudaFree(device_ptr);
  }
  void* get() const { return device_ptr; }

 private:
  void* device_ptr;
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
    fprintf(stderr, "No %s devices found\n", GPUPLATFORM);
    return;
  }

  for (int iDevice = 0; iDevice < deviceCount; ++iDevice) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, iDevice);
    printf("%s Device %d: %s\n", GPUPLATFORM, iDevice, deviceProp.name);
  }
}

using MatSym3DGPU = o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepSymGPU<float, 3>>;
using MatSym3D = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
using Mat3DGPU = o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepStdGPU<float, 3, 3>>;
using Mat3D = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepStd<float, 3, 3>>;

template <typename MatrixType>
__device__ void printMatrix(const MatrixType& matrix, const char* name)
{
  printf("%s(0,0) = %f, %s(0,1) = %f, %s(0,2) = %f\n", name, matrix(0, 0), name, matrix(0, 1), name, matrix(0, 2));
  printf("%s(1,0) = %f, %s(1,1) = %f, %s(1,2) = %f\n", name, matrix(1, 0), name, matrix(1, 1), name, matrix(1, 2));
  printf("%s(2,0) = %f, %s(2,1) = %f, %s(2,2) = %f\n", name, matrix(2, 0), name, matrix(2, 1), name, matrix(2, 2));
}

// Invert test for symmetric matrix
template <typename T, int D>
__global__ void invertSymMatrixKernel(MatSym3DGPU* matrix)
{
  MatSym3DGPU smat2 = *matrix;

  printMatrix(*matrix, "A");
  printMatrix(smat2, "B");

  printf("\nInverting A...\n");
  matrix->Invert();

  printMatrix(*matrix, "A");

  printf("\nC = (A^-1) * B...\n");
  auto smat3 = (*matrix) * smat2;

  printMatrix(smat3, "C");

  printf("\nEvaluating...\n");
  MatSym3DGPU tmp;
  o2::math_utils::AssignSym::Evaluate(tmp, smat3);

  printMatrix(tmp, "A");
  *matrix = tmp;
}

// Invert test for general matrix
template <typename T, int D>
__global__ void invertMatrixKernel(Mat3DGPU* matrix)
{
  Mat3DGPU smat2 = *matrix;

  printMatrix(*matrix, "A");
  printMatrix(smat2, "B");

  printf("\nInverting A...\n");
  matrix->Invert();

  printMatrix(*matrix, "A");

  printf("\nC = (A^-1) * B...\n");
  auto smat3 = (*matrix) * smat2;

  printMatrix(smat3, "C");

  printf("\nEvaluating...\n");
  Mat3DGPU tmp;
  o2::math_utils::Assign::Evaluate(tmp, smat3);

  printMatrix(tmp, "A");
  *matrix = tmp;
}

struct GPUSMatrixImplFixture {
  GPUSMatrixImplFixture() : i(3), SMatrixSym3D_d(sizeof(MatSym3DGPU)), SMatrixSym3D_h(), SMatrix3D_d(sizeof(Mat3DGPU)), SMatrix3D_h()
  {
    prologue();
    initializeMatrices();
    printMatrixSizes();
  }

  ~GPUSMatrixImplFixture() = default;

  void initializeMatrices()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 10.0);

    // Initialize host matrices with random values
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        SMatrixSym3D_h(i, j) = dis(gen);
        if (i != j) {
          SMatrixSym3D_h(j, i) = SMatrixSym3D_h(i, j); // Ensure symmetry
        }
        SMatrix3D_h(i, j) = dis(gen);
        if (i != j) {
          SMatrix3D_h(j, i) = dis(gen);
        }
      }
    }

    // Copy host matrices to device
    CUDA_CHECK(cudaMemcpy(SMatrixSym3D_d.get(), &SMatrixSym3D_h, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(SMatrix3D_d.get(), &SMatrix3D_h, sizeof(Mat3DGPU), cudaMemcpyHostToDevice));
  }

  void printMatrixSizes() const
  {
    printf("sizeof(MatSym3DGPU) = %zu\n", sizeof(MatSym3DGPU));
    printf("sizeof(MatSym3D) = %zu\n", sizeof(MatSym3D));
    printf("sizeof(Mat3DGPU) = %zu\n", sizeof(Mat3DGPU));
    printf("sizeof(Mat3D) = %zu\n", sizeof(Mat3D));
  }

  int i;
  CudaMemory SMatrixSym3D_d;
  MatSym3D SMatrixSym3D_h;
  CudaMemory SMatrix3D_d;
  Mat3D SMatrix3D_h;
};

BOOST_FIXTURE_TEST_CASE(DummyFixtureUsage, GPUSMatrixImplFixture)
{
  invertSymMatrixKernel<float, 3><<<1, 1>>>(static_cast<MatSym3DGPU*>(SMatrixSym3D_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrixSym3D_h, SMatrixSym3D_d.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  MatSym3D identitySym;
  identitySym(0, 0) = 1;
  identitySym(1, 1) = 1;
  identitySym(2, 2) = 1;
  BOOST_TEST(SMatrixSym3D_h == identitySym);

  invertMatrixKernel<float, 3><<<1, 1>>>(static_cast<Mat3DGPU*>(SMatrix3D_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrix3D_h, SMatrix3D_d.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  Mat3D identity;
  identity(0, 0) = 1;
  identity(1, 1) = 1;
  identity(2, 2) = 1;
  BOOST_TEST(SMatrix3D_h == identity);
}

// Transpose test for symmetric matrix
template <typename T>
__global__ void testSymTransposeTwiceKernel(MatSym3DGPU* matrix)
{
  auto transposedOnce = o2::math_utils::Transpose(*matrix);
  auto transposedTwice = o2::math_utils::Transpose(transposedOnce);

  *matrix = transposedTwice;
}

// Transpose test for general matrix
template <typename T>
__global__ void testTransposeTwiceKernel(Mat3DGPU* matrix)
{
  auto transposedOnce = o2::math_utils::Transpose(*matrix);
  auto transposedTwice = o2::math_utils::Transpose(transposedOnce);

  *matrix = transposedTwice;
}

BOOST_FIXTURE_TEST_CASE(TestMatrixDoubleTranspose, GPUSMatrixImplFixture)
{
  testSymTransposeTwiceKernel<<<1, 1>>>(static_cast<MatSym3DGPU*>(SMatrixSym3D_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrixSym3D_h, SMatrixSym3D_d.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(SMatrixSym3D_h(i, j) == (i * 3 + j + 1));
    }
  }

  testTransposeTwiceKernel<<<1, 1>>>(static_cast<Mat3DGPU*>(SMatrix3D_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrix3D_h, SMatrix3D_d.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(SMatrix3D_h(i, j) == (i * 3 + j + 1));
    }
  }

  // Test on CPU for symmetric matrix
  MatSym3D cpuSymMatrix = SMatrixSym3D_h;
  MatSym3D transposedSymOnce = ROOT::Math::Transpose(cpuSymMatrix);
  MatSym3D transposedSymTwice = ROOT::Math::Transpose(transposedSymOnce);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(cpuSymMatrix(i, j) == transposedSymTwice(i, j));
    }
  }

  // Test on CPU for general matrix
  Mat3D cpuMatrix = SMatrix3D_h;
  Mat3D transposedOnce = ROOT::Math::Transpose(cpuMatrix);
  Mat3D transposedTwice = ROOT::Math::Transpose(transposedOnce);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(cpuMatrix(i, j) == transposedTwice(i, j));
    }
  }
}

// Multiplication test for symmetric matrix
template <typename T>
__global__ void testSymMatrixMultiplicationKernel(
  MatSym3DGPU* matrixA,
  MatSym3DGPU* matrixB,
  MatSym3DGPU* result)
{
  *result = (*matrixA) * (*matrixB);
}

// Multiplication test for general matrix
template <typename T>
__global__ void testMatrixMultiplicationKernel(
  Mat3DGPU* matrixA,
  Mat3DGPU* matrixB,
  Mat3DGPU* result)
{
  *result = (*matrixA) * (*matrixB);
}

BOOST_FIXTURE_TEST_CASE(TestMatrixMultiplication, GPUSMatrixImplFixture)
{
  MatSym3DGPU matrixSymB_h;
  MatSym3D resultSym_h;
  CudaMemory matrixSymB_d(sizeof(MatSym3DGPU));
  CudaMemory resultSym_d(sizeof(MatSym3DGPU));

  // Initialize matrixSymB_h with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0, 10.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      matrixSymB_h(i, j) = dis(gen);
      if (i != j) {
        matrixSymB_h(j, i) = matrixSymB_h(i, j); // Ensure symmetry
      }
    }
  }

  CUDA_CHECK(cudaMemcpy(matrixSymB_d.get(), &matrixSymB_h, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));

  testSymMatrixMultiplicationKernel<<<1, 1>>>(static_cast<MatSym3DGPU*>(SMatrixSym3D_d.get()), static_cast<MatSym3DGPU*>(matrixSymB_d.get()), static_cast<MatSym3DGPU*>(resultSym_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&resultSym_h, resultSym_d.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  // Perform the same matrix multiplication on CPU for comparison
  MatSym3D resultSym_cpu = SMatrixSym3D_h * matrixSymB_h;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(resultSym_h(i, j) == resultSym_cpu(i, j), boost::test_tools::tolerance(0.00001f));
    }
  }

  Mat3DGPU matrixB_h;
  Mat3D result_h;
  CudaMemory matrixB_d(sizeof(Mat3DGPU));
  CudaMemory result_d(sizeof(Mat3DGPU));

  // Initialize matrixB_h with random values
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      matrixB_h(i, j) = dis(gen);
    }
  }

  CUDA_CHECK(cudaMemcpy(matrixB_d.get(), &matrixB_h, sizeof(Mat3DGPU), cudaMemcpyHostToDevice));

  testMatrixMultiplicationKernel<<<1, 1>>>(static_cast<Mat3DGPU*>(SMatrix3D_d.get()), static_cast<Mat3DGPU*>(matrixB_d.get()), static_cast<Mat3DGPU*>(result_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&result_h, result_d.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  // Perform the same matrix multiplication on CPU for comparison
  Mat3D result_cpu = SMatrix3D_h * matrixB_h;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(result_h(i, j) == result_cpu(i, j));
    }
  }
}

// Copy test for symmetric matrix
template <typename T>
__global__ void copySymMatrixKernel(
  const MatSym3DGPU* srcMatrix,
  MatSym3DGPU* dstMatrix)
{
  *dstMatrix = *srcMatrix;
}

// Copy test for general matrix
template <typename T>
__global__ void copyMatrixKernel(
  const Mat3DGPU* srcMatrix,
  Mat3DGPU* dstMatrix)
{
  *dstMatrix = *srcMatrix;
}

BOOST_FIXTURE_TEST_CASE(TestMatrixCopyingAndComparison, GPUSMatrixImplFixture)
{
  MatSym3DGPU copiedSymMatrix_h;
  CudaMemory copiedSymMatrix_d(sizeof(MatSym3DGPU));

  copySymMatrixKernel<<<1, 1>>>(static_cast<MatSym3DGPU*>(SMatrixSym3D_d.get()), static_cast<MatSym3DGPU*>(copiedSymMatrix_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&copiedSymMatrix_h, copiedSymMatrix_d.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(SMatrixSym3D_h(i, j) == copiedSymMatrix_h(i, j), boost::test_tools::tolerance(0.00001f));
    }
  }

  Mat3DGPU copiedMatrix_h;
  CudaMemory copiedMatrix_d(sizeof(Mat3DGPU));

  copyMatrixKernel<<<1, 1>>>(static_cast<Mat3DGPU*>(SMatrix3D_d.get()), static_cast<Mat3DGPU*>(copiedMatrix_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&copiedMatrix_h, copiedMatrix_d.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      BOOST_TEST(SMatrix3D_h(i, j) == copiedMatrix_h(i, j), boost::test_tools::tolerance(0.00001f));
    }
  }
}
