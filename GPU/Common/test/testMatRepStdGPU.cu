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
#include <cuda.h>
#endif

#include <boost/test/unit_test.hpp>
#include <iostream>

#include <MathUtils/SMatrixGPU.h>
#include <Math/SMatrix.h>

template <typename T>
void discardResult(const T&) {}

void prologue() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || !deviceCount) {
        std::cerr << "No " << GPUPLATFORM << " devices found" << std::endl;
        return;
    }

    for (int iDevice = 0; iDevice < deviceCount; ++iDevice) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, iDevice);
        std::cout << GPUPLATFORM << " Device " << iDevice << ": " << deviceProp.name << std::endl;
    }
}

using Mat3DGPU = o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepStdGPU<float, 3, 3>>;
using Mat3D = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepStd<float, 3, 3>>;

template <typename MatrixType>
__device__ void printMatrix(const MatrixType& matrix, const char* name) {
    printf("%s(0,0) = %f, %s(0,1) = %f, %s(0,2) = %f\n", name, matrix(0, 0), name, matrix(0, 1), name, matrix(0, 2));
    printf("%s(1,0) = %f, %s(1,1) = %f, %s(1,2) = %f\n", name, matrix(1, 0), name, matrix(1, 1), name, matrix(1, 2));
    printf("%s(2,0) = %f, %s(2,1) = %f, %s(2,2) = %f\n", name, matrix(2, 0), name, matrix(2, 1), name, matrix(2, 2));
}

// Invert test
template <typename T, int D>
__global__ void invertMatrixKernel(o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepStdGPU<float, 3, 3>>* matrix) {
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
    GPUSMatrixImplFixture() : i(3), SMatrix3D_d(nullptr), SMatrix3D_h() {
        prologue();
        SMatrix3D_h(0, 0) = 1;
        SMatrix3D_h(1, 1) = 2;
        SMatrix3D_h(2, 2) = 3;
        SMatrix3D_h(0, 1) = 4;
        SMatrix3D_h(0, 2) = 5;
        SMatrix3D_h(1, 2) = 6;

        cudaError_t error = cudaMalloc(&SMatrix3D_d, sizeof(Mat3DGPU));
        if (error != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
        }

        error = cudaMemcpy(SMatrix3D_d, &SMatrix3D_h, sizeof(Mat3DGPU), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(error) << std::endl;
        }

        std::cout << "sizeof(Mat3DGPU) = " << sizeof(Mat3DGPU) << std::endl;
        std::cout << "sizeof(Mat3D) = " << sizeof(Mat3D) << std::endl;
    }

    ~GPUSMatrixImplFixture() {
        cudaFree(SMatrix3D_d);
    }

    int i;
    Mat3DGPU* SMatrix3D_d; // device ptr
    Mat3D SMatrix3D_h;
};

BOOST_FIXTURE_TEST_CASE(DummyFixtureUsage, GPUSMatrixImplFixture) {
    invertMatrixKernel<float, 3><<<1, 1>>>(SMatrix3D_d);
    cudaDeviceSynchronize();

    cudaMemcpy(&SMatrix3D_h, SMatrix3D_d, sizeof(Mat3DGPU), cudaMemcpyDeviceToHost);

    Mat3D identity;
    identity(0, 0) = 1;
    identity(1, 1) = 1;
    identity(2, 2) = 1;
    BOOST_TEST(SMatrix3D_h == identity);
}

// Transpose test
template <typename T>
__global__ void testTransposeTwiceKernel(o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepStdGPU<float, 3, 3>>* matrix) {
    auto transposedOnce = o2::math_utils::Transpose(*matrix);
    auto transposedTwice = o2::math_utils::Transpose(transposedOnce);

    *matrix = transposedTwice;
}

BOOST_FIXTURE_TEST_CASE(TestMatrixDoubleTranspose, GPUSMatrixImplFixture) {
    testTransposeTwiceKernel<<<1, 1>>>(SMatrix3D_d);
    cudaDeviceSynchronize();
    cudaError_t error = cudaMemcpy(&SMatrix3D_h, SMatrix3D_d, sizeof(Mat3DGPU), cudaMemcpyDeviceToHost);
    BOOST_REQUIRE(error == cudaSuccess);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            BOOST_TEST(SMatrix3D_h(i, j) == (i * 3 + j + 1));
        }
    }

    // Test on CPU
    Mat3D cpuMatrix = SMatrix3D_h;
    Mat3D transposedOnce = ROOT::Math::Transpose(cpuMatrix);
    Mat3D transposedTwice = ROOT::Math::Transpose(transposedOnce);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            BOOST_TEST(cpuMatrix(i, j) == transposedTwice(i, j));
        }
    }
}

// Multiplication test
template <typename T>
__global__ void testMatrixMultiplicationKernel(
    o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepStdGPU<float, 3, 3>>* matrixA,
    o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepStdGPU<float, 3, 3>>* matrixB,
    o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepStdGPU<float, 3, 3>>* result)
{
    *result = (*matrixA) * (*matrixB);
}

BOOST_FIXTURE_TEST_CASE(TestMatrixMultiplication, GPUSMatrixImplFixture) {
    Mat3DGPU *matrixB_d, *result_d;
    Mat3D matrixB_h, result_h;

    matrixB_h(0, 0) = 9; matrixB_h(0, 1) = 8; matrixB_h(0, 2) = 7;
    matrixB_h(1, 0) = 6; matrixB_h(1, 1) = 5; matrixB_h(1, 2) = 4;
    matrixB_h(2, 0) = 3; matrixB_h(2, 1) = 2; matrixB_h(2, 2) = 1;

    cudaMalloc(&matrixB_d, sizeof(Mat3DGPU));
    cudaMalloc(&result_d, sizeof(Mat3DGPU));
    cudaMemcpy(matrixB_d, &matrixB_h, sizeof(Mat3DGPU), cudaMemcpyHostToDevice);

    testMatrixMultiplicationKernel<<<1, 1>>>(SMatrix3D_d, matrixB_d, result_d);
    cudaDeviceSynchronize();

    cudaMemcpy(&result_h, result_d, sizeof(Mat3DGPU), cudaMemcpyDeviceToHost);

    Mat3D expected;
    expected(0, 0) = 26; expected(0, 1) = 20; expected(0, 2) = 14;
    expected(1, 0) = 57; expected(1, 1) = 44; expected(1, 2) = 31;
    expected(2, 0) = 96; expected(2, 1) = 74; expected(2, 2) = 52;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            BOOST_TEST(result_h(i, j) == expected(i, j));
        }
    }

    cudaFree(matrixB_d);
    cudaFree(result_d);
}
