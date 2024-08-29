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

#ifdef __HIPCC__
#include "hip/hip_runtime.h"
#else
#include <cuda.h>
#endif

#include "GPUCommonDef.h"
#include "DCAFitter/DCAFitterN.h"

#include "DCAFitterNKernels.h"

using DCAFitter2 = o2::vertexing::DCAFitterN<2, o2::track::TrackParCov>;
using DCAFitter3 = o2::vertexing::DCAFitterN<3, o2::track::TrackParCov>;

namespace o2::vertexing
{
namespace gpu
{
GPUg() void testKernel()
{
  DCAFitter2 ft2;
  DCAFitter3 ft3;
  o2::track::TrackParCov tr;
  ft2.process(tr, tr);
  ft3.process(tr, tr, tr);
}
} // namespace gpu

void kernelHandler()
{
  gpu::testKernel<<<1, 1>>>();
}
} // namespace o2::vertexing