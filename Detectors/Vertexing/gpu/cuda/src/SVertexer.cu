// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if defined(__HIPCC__)
#include "DetectorsVertexingHIP/SVertexer.h"
#else
#include "DetectorsVertexingCUDA/SVertexer.h"
#endif

// #include "DetectorsVertexing/SMatrixGPU.h"
#include "DetectorsVertexing/DCAFitterN.h" // <- target

namespace o2
{
namespace vertexing
{

using Vec3D = o2::math_utils::SVector<double, 3>;
using MatSym3D = o2::math_utils::SMatrix<double, 3, 3, o2::math_utils::MatRepSym<double, 3>>;
using MatStd3D = o2::math_utils::SMatrix<double, 3, 3, o2::math_utils::MatRepStd<double, 3>>;

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

// Kernels
GPUg() void helloKernel()
{
  o2::vertexing::DCAFitterN<2> mFitter2Prong;
  o2::vertexing::DCAFitterN<3> mFitter3Prong;
  
  o2::gpu::gpustd::array<Vec3D, 2> arrVectors;
  o2::math_utils::MatRepSym<double, 3> repsym;
  o2::math_utils::MatRepStd<double, 3> repstd;
  MatSym3D matSimRepBased;
  // o2::math_utils::SMatrixIdentity id;
  // MatStd3D matStdRepBased{id};
  MatStd3D matStdRepBased;
  for (size_t iA{0}; iA < 3; ++iA) {
    for (size_t iV{0}; iV < 3; ++iV) {
      matStdRepBased(iA, iV) = 1.f;
    }
  }

  MatStd3D result;
  matSimRepBased(0, 0) = 1.0f;
  matSimRepBased(2, 2) = 1.0f;
  matSimRepBased(0, 2) = 1.0f;
  matSimRepBased(0, 1) = 1.0f;
  matSimRepBased(1, 1) = 1.0f;
  matSimRepBased(2, 1) = 1.0f;

  result = matSimRepBased * matStdRepBased;
  // matStdRepBased(1, 1) = 4.5f;
  // matStdRepBased(2, 1) = 9.9f;

  for (size_t iA{0}; iA < 2; ++iA) {
    for (size_t iV{0}; iV < 3; ++iV) {
      arrVectors[iA][iV] = 3.f;
    }
  }

  auto res = o2::math_utils::Dot(arrVectors[0], arrVectors[1]);

  // Debug
  printf("Initialisation result:\n");
  for (size_t iA{0}; iA < 2; ++iA) {
    for (size_t iV{0}; iV < 3; ++iV) {
      printf("(%lu, %lu): %f \t", iA, iV, arrVectors[iA][iV]);
    }
    printf("\n");
  }

  printf("Dot result %f\n", res);
  printf(" = == = = = = == == == =\n");
  printf("Sym representation:\n");
  for (size_t iA{0}; iA < 3; ++iA) {
    for (size_t iV{0}; iV < 3; ++iV) {
      printf("(%lu, %lu): %f \t", iA, iV, matSimRepBased(iA, iV));
    }
    printf("\n");
  }

  printf("\n\nStd representation:\n");
  for (size_t iA{0}; iA < 3; ++iA) {
    for (size_t iV{0}; iV < 3; ++iV) {
      printf("(%lu, %lu): %f \t", iA, iV, matStdRepBased(iA, iV));
    }
    printf("\n");
  }

  printf("\n\nResult sym*std:\n");
  for (size_t iA{0}; iA < 3; ++iA) {
    for (size_t iV{0}; iV < 3; ++iV) {
      printf("(%lu, %lu): %f \t", iA, iV, result(iA, iV));
    }
    printf("\n");
  }
  printf("\n\n");
}

void hello_util()
{
  helloKernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  cudaCheckError();
}

} // namespace vertexing
} // namespace o2