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

#include <iostream>
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
    CUDA_CHECK(cudaFree(device_ptr));
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
    std::cerr << "No " << GPUPLATFORM << " devices found" << std::endl;
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

// Function to compare two matrices element-wise with a specified tolerance
template <typename MatrixType>
void compareMatrices(const MatrixType& mat1, const MatrixType& mat2, float tolerance)
{
  auto tol = boost::test_tools::tolerance(tolerance);

  for (unsigned int i = 0; i < mat1.kRows; ++i) {
    for (unsigned int j = 0; j < mat1.kCols; ++j) {
      BOOST_TEST(mat1(i, j) == mat2(i, j), tol);
    }
  }
}

// Invert test for symmetric matrix
template <typename T, int D>
__global__ void invertSymMatrixKernel(MatSym3DGPU* matrix)
{
  printf("\nStart inverting symmetric matrix\n");
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
  printf("\n-------------------------------------------------------\n");
}

// Invert test for general matrix
template <typename T, int D>
__global__ void invertMatrixKernel(Mat3DGPU* matrix)
{
  printf("\nStart inverting general matrix\n");
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
  o2::math_utils::Assign<float, 3, 3, decltype(smat3), o2::math_utils::MatRepStdGPU<float, 3, 3>, o2::math_utils::MatRepStdGPU<float, 3, 3>>::Evaluate(tmp, smat3);

  printMatrix(tmp, "A");
  *matrix = tmp;
  printf("\n-------------------------------------------------------\n");
}

struct GPUSMatrixImplFixtureSolo {
  GPUSMatrixImplFixtureSolo() : i(3), SMatrixSym3D_d(sizeof(MatSym3DGPU)), SMatrixSym3D_h(), SMatrix3D_d(sizeof(Mat3DGPU)), SMatrix3D_h()
  {
    prologue();
    initializeMatrices();
    printMatrixSizes();
  }

  ~GPUSMatrixImplFixtureSolo() = default;

  void initializeMatrices()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 10.0);

    // Initialize host matrices with random values
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        SMatrixSym3D_h(i, j) = dis(gen);
        SMatrix3D_h(i, j) = dis(gen);
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

BOOST_FIXTURE_TEST_CASE(MatrixInversion, GPUSMatrixImplFixtureSolo)
{
  float tolerance = 0.00001f;

  invertSymMatrixKernel<float, 3><<<1, 1>>>(static_cast<MatSym3DGPU*>(SMatrixSym3D_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrixSym3D_h, SMatrixSym3D_d.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  MatSym3D identitySym;
  identitySym(0, 0) = 1;
  identitySym(1, 1) = 1;
  identitySym(2, 2) = 1;
  compareMatrices(SMatrixSym3D_h, identitySym, tolerance);

  invertMatrixKernel<float, 3><<<1, 1>>>(static_cast<Mat3DGPU*>(SMatrix3D_d.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrix3D_h, SMatrix3D_d.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  Mat3D identity;
  identity(0, 0) = 1;
  identity(1, 1) = 1;
  identity(2, 2) = 1;
  compareMatrices(SMatrix3D_h, identity, tolerance);
}

struct GPUSMatrixImplFixtureDuo {
  GPUSMatrixImplFixtureDuo() : i(3), SMatrixSym3D_d_A(sizeof(MatSym3DGPU)), SMatrixSym3D_h_A(), SMatrix3D_d_A(sizeof(Mat3DGPU)), SMatrix3D_h_A(), SMatrixSym3D_d_B(sizeof(MatSym3DGPU)), SMatrixSym3D_h_B(), SMatrix3D_d_B(sizeof(Mat3DGPU)), SMatrix3D_h_B()
  {
    prologue();
    initializeMatrices();
    printMatrixSizes();
  }

  ~GPUSMatrixImplFixtureDuo() = default;

  void initializeMatrices()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0, 10.0);

    // Initialize host matrices with random values
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        SMatrixSym3D_h_A(i, j) = dis(gen);
        SMatrix3D_h_A(i, j) = dis(gen);

        SMatrixSym3D_h_B(i, j) = dis(gen);
        SMatrix3D_h_B(i, j) = dis(gen);
      }
    }

    // Copy host matrices to device
    CUDA_CHECK(cudaMemcpy(SMatrixSym3D_d_A.get(), &SMatrixSym3D_h_A, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(SMatrix3D_d_A.get(), &SMatrix3D_h_A, sizeof(Mat3DGPU), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(SMatrixSym3D_d_B.get(), &SMatrixSym3D_h_B, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(SMatrix3D_d_B.get(), &SMatrix3D_h_B, sizeof(Mat3DGPU), cudaMemcpyHostToDevice));
  }

  void printMatrixSizes() const
  {
    printf("sizeof(MatSym3DGPU) = %zu\n", sizeof(MatSym3DGPU));
    printf("sizeof(MatSym3D) = %zu\n", sizeof(MatSym3D));
    printf("sizeof(Mat3DGPU) = %zu\n", sizeof(Mat3DGPU));
    printf("sizeof(Mat3D) = %zu\n", sizeof(Mat3D));
  }

  int i;
  CudaMemory SMatrixSym3D_d_A;
  MatSym3D SMatrixSym3D_h_A;

  CudaMemory SMatrixSym3D_d_B;
  MatSym3D SMatrixSym3D_h_B;

  CudaMemory SMatrix3D_d_A;
  Mat3D SMatrix3D_h_A;

  CudaMemory SMatrix3D_d_B;
  Mat3D SMatrix3D_h_B;
};

// Copy test for symmetric matrix
template <typename T>
__global__ void copySymMatrixKernel(
  MatSym3DGPU* srcMatrix,
  MatSym3DGPU* dstMatrix)
{
  printf("\nStart copying general matrix\n");
  printMatrix(*dstMatrix, "Before copying: ");
  printf("\nCopied values:\n");
  printMatrix(*srcMatrix, "Copied values: ");
  printf("\nResult:\n");
  *dstMatrix = *srcMatrix;
  printMatrix(*dstMatrix, "After copying: ");
  printf("\n-------------------------------------------------------\n");
}

// Copy test for general matrix
template <typename T>
__global__ void copyMatrixKernel(
  Mat3DGPU* srcMatrix,
  Mat3DGPU* dstMatrix)
{
  printf("\nStart copying general matrix\n");
  printMatrix(*dstMatrix, "Before copying: ");
  printf("\nCopied values:\n");
  printMatrix(*srcMatrix, "Copied values: ");
  printf("\nResult:\n");
  *dstMatrix = *srcMatrix;
  printMatrix(*dstMatrix, "After copying: ");
  printf("\n-------------------------------------------------------\n");
}

BOOST_FIXTURE_TEST_CASE(TestMatrixCopyingAndComparison, GPUSMatrixImplFixtureDuo)
{
  copySymMatrixKernel<float><<<1, 1>>>(static_cast<MatSym3DGPU*>(SMatrixSym3D_d_A.get()), static_cast<MatSym3DGPU*>(SMatrixSym3D_d_B.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrixSym3D_h_B, SMatrixSym3D_d_B.get(), sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  compareMatrices(SMatrixSym3D_h_A, SMatrixSym3D_h_B, 0.0);

  copyMatrixKernel<float><<<1, 1>>>(static_cast<Mat3DGPU*>(SMatrix3D_d_A.get()), static_cast<Mat3DGPU*>(SMatrix3D_d_B.get()));
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&SMatrix3D_h_B, SMatrix3D_d_B.get(), sizeof(Mat3DGPU), cudaMemcpyDeviceToHost));

  compareMatrices(SMatrix3D_h_A, SMatrix3D_h_B, 0.0);
}
