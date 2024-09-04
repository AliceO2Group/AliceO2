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
//
/// \author matteo.concas@cern.ch
#define BOOST_TEST_MODULE Test DCAFitterN class on GPU

#ifdef __HIPCC__
#define GPUPLATFORM "HIP"
#include "hip/hip_runtime.h"
#else
#define GPUPLATFORM "CUDA"
#include <cuda.h>
#endif

#include "DCAFitter/DCAFitterN.h"
#include "GPUCommonDef.h"
#include <boost/test/unit_test.hpp>

namespace gpu
{
GPUg() void testDCAFitterInstanceKernel()
{
  o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
  ft.setBz(0.f);
}
} // namespace gpu
BOOST_AUTO_TEST_CASE(DCAFitterNProngs)
{
}