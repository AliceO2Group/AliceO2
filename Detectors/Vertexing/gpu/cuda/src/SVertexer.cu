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
GPUg() void testFitterKernel()
{
  double bz = 5.0;
  o2::vertexing::DCAFitterN<2> mFitter2Prong;
  mFitter2Prong.setBz(bz);
  mFitter2Prong.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
  mFitter2Prong.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
  mFitter2Prong.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
  mFitter2Prong.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
  mFitter2Prong.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

  printf("End.\n");
}

void hello_util()
{
  testFitterKernel<<<1, 1>>>();
  cudaCheckError();
}

} // namespace vertexing
} // namespace o2
