// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsVertexingCUDA/SVertexer.h"
// #include "DetectorsVertexing/DCAFitterN.h" // <- target
// #include "DetectorsVertexing/HelixHelper.h"
// #include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace vertexing
{
namespace gpu
{ // Kernels
GPUg() void helloKernel() {
    o2::vertexing::DCAFitterN<2> mFitter2Prong;
    printf("Hello World from GPU!\n");
}
}

// Steers
void SVertexerGPU::hello()
{
    gpu::helloKernel<<<1,1>>>();
}

}
}