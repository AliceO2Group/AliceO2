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

#include "DetectorsVertexing/SMatrixGPU.h"
// #include "DetectorsVertexing/DCAFitterN.h" // <- target
// #include "DetectorsVertexing/HelixHelper.h"
// #include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace vertexing
{

using Vec3D = o2::math_utils::SVector<double, 3>;

// Kernels
GPUg() void helloKernel()
{
  // o2::vertexing::DCAFitterN<2> mFitter2Prong;
  o2::gpu::gpustd::array<Vec3D, 2> mPCA;
  printf("Hello World from GPU!\n");
}

void hello_util()
{
  helloKernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

} // namespace vertexing
} // namespace o2