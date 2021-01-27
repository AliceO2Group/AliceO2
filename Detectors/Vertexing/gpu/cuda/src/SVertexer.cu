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

// #include "MathUtils/SMatrixGPU.h"
#include "DetectorsVertexing/DCAFitterN.h" // <- target
// #include "DetectorsVertexing/HelixHelper.h"
// #include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace vertexing
{
namespace gpu
{

// Kernels
GPUg() void helloKernel()
{
  o2::vertexing::DCAFitterN<2> mFitter2Prong;
  int arr[3] = {1, 2, 3};
  printf("Hello World from GPU!\n");
}

} // namespace gpu

void hello_util()
{
  gpu::helloKernel<<<1, 1>>>();
}

} // namespace vertexing
} // namespace o2