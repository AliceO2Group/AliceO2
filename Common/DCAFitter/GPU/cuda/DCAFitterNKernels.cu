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

namespace o2::vertexing::gpu
{
GPUg() void printKernel(o2::vertexing::DCAFitterN<2>* ft)
{
  printf("hello world");
  // if (threadIdx.x == 0) {
  //   ft->print();
  // }
}
GPUg() void processKernel(o2::vertexing::DCAFitterN<2>* ft, o2::track::TrackParCov* t1, o2::track::TrackParCov* t2, int* res)
{
  *res = 30000;/*ft->process(*t1, *t2);*/
}
} // namespace o2::vertexing::gpu