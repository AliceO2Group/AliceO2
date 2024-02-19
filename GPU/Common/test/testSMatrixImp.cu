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
void discardResult(const T&)
{
}

void prologue()
{
  int deviceCount;
  discardResult(cudaGetDeviceCount(&deviceCount));
  if (!deviceCount) {
    std::cerr << "No " << GPUPLATFORM << " devices found" << std::endl;
  }
  for (int iDevice = 0; iDevice < deviceCount; ++iDevice) {
    cudaDeviceProp deviceProp;
    discardResult(cudaGetDeviceProperties(&deviceProp, iDevice));
    std::cout << GPUPLATFORM << " Device " << iDevice << ": " << deviceProp.name << std::endl;
  }
}

using MatSym3DGPU = o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepSymGPU<float, 3>>;
using MatSym3D = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;

template <typename T, int D>
__global__ void invertSymMatrixKernel(o2::math_utils::SMatrixGPU<float, 3, 3, o2::math_utils::MatRepSymGPU<float, 3>>* matrix)
{
  MatSym3DGPU smat2 = *matrix;

  printf("A(0,0) = %f, A(0,1) = %f, A(0,2) = %f\n", (*matrix)(0, 0), (*matrix)(0, 1), (*matrix)(0, 2));
  printf("A(1,0) = %f, A(1,1) = %f, A(1,2) = %f\n", (*matrix)(1, 0), (*matrix)(1, 1), (*matrix)(1, 2));
  printf("A(2,0) = %f, A(2,1) = %f, A(2,2) = %f\n", (*matrix)(2, 0), (*matrix)(2, 1), (*matrix)(2, 2));

  printf("B(0,0) = %f, B(0,1) = %f, B(0,2) = %f\n", smat2(0, 0), smat2(0, 1), smat2(0, 2));
  printf("B(1,0) = %f, B(1,1) = %f, B(1,2) = %f\n", smat2(1, 0), smat2(1, 1), smat2(1, 2));
  printf("B(2,0) = %f, B(2,1) = %f, B(2,2) = %f\n", smat2(2, 0), smat2(2, 1), smat2(2, 2));

  printf("\nInverting A...\n");
  matrix->Invert();

  printf("A(0,0) = %f, A(0,1) = %f, A(0,2) = %f\n", (*matrix)(0, 0), (*matrix)(0, 1), (*matrix)(0, 2));
  printf("A(1,0) = %f, A(1,1) = %f, A(1,2) = %f\n", (*matrix)(1, 0), (*matrix)(1, 1), (*matrix)(1, 2));
  printf("A(2,0) = %f, A(2,1) = %f, A(2,2) = %f\n", (*matrix)(2, 0), (*matrix)(2, 1), (*matrix)(2, 2));

  printf("\nC = (A^-1) * B...\n");
  auto smat3 = (*matrix) * smat2;

  printf("C(0,0) = %f, C(0,1) = %f, C(0,2) = %f\n", smat3(0, 0), smat3(0, 1), smat3(0, 2));
  printf("C(1,0) = %f, C(1,1) = %f, C(1,2) = %f\n", smat3(1, 0), smat3(1, 1), smat3(1, 2));
  printf("C(2,0) = %f, C(2,1) = %f, C(2,2) = %f\n", smat3(2, 0), smat3(2, 1), smat3(2, 2));

  printf("\nEvaluating...\n");
  MatSym3DGPU tmp;
  o2::math_utils::AssignSym::Evaluate(tmp, smat3);

  printf("A(0,0) = %f, A(0,1) = %f, A(0,2) = %f\n", tmp(0, 0), tmp(0, 1), tmp(0, 2));
  printf("A(1,0) = %f, A(1,1) = %f, A(1,2) = %f\n", tmp(1, 0), tmp(1, 1), tmp(1, 2));
  printf("A(2,0) = %f, A(2,1) = %f, A(2,2) = %f\n", tmp(2, 0), tmp(2, 1), tmp(2, 2));
  (*matrix) = tmp;
}

struct GPUSMatrixImplFixture {
  GPUSMatrixImplFixture() : SMatrix3D_d(nullptr)
  {
    prologue();

    SMatrix3D_h(0, 0) = 1;
    SMatrix3D_h(1, 1) = 2;
    SMatrix3D_h(2, 2) = 3;
    SMatrix3D_h(0, 1) = 4;
    SMatrix3D_h(0, 2) = 5;
    SMatrix3D_h(1, 2) = 6;

    discardResult(cudaMalloc(&SMatrix3D_d, sizeof(MatSym3DGPU)));
    discardResult(cudaMemcpy(SMatrix3D_d, &SMatrix3D_h, sizeof(MatSym3DGPU), cudaMemcpyHostToDevice));

    std::cout << "sizeof(MatSym3DGPU) = " << sizeof(MatSym3DGPU) << std::endl;
    std::cout << "sizeof(MatSym3D) = " << sizeof(MatSym3D) << std::endl;
    i = 3;
  }

  ~GPUSMatrixImplFixture()
  {
    discardResult(cudaFree(SMatrix3D_d));
  }

  int i;
  MatSym3DGPU* SMatrix3D_d; // device ptr
  MatSym3D SMatrix3D_h;
};

BOOST_FIXTURE_TEST_CASE(DummyFixtureUsage, GPUSMatrixImplFixture)
{
  invertSymMatrixKernel<float, 3><<<1, 1>>>(SMatrix3D_d);
  discardResult(cudaDeviceSynchronize());

  discardResult(cudaMemcpy(&SMatrix3D_h, SMatrix3D_d, sizeof(MatSym3DGPU), cudaMemcpyDeviceToHost));

  MatSym3D identity;
  identity(0, 0) = 1;
  identity(1, 1) = 1;
  identity(2, 2) = 1;
  BOOST_TEST(SMatrix3D_h == identity);
}