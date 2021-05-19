// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file GPUbenchmark.cu
/// \author: mconcas@cern.ch

#include <GPUbenchmark.h>
#include <stdio.h>

namespace o2
{
namespace benchmark
{
namespace gpu
{
GPUg() void helloKernel()
{
  printf("Hello World from GPU!!\n");
}
} // namespace gpu

void hello_util()
{
  gpu::helloKernel<<<1, 1>>>();
}
} // namespace benchmark
} // namespace o2