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
///

#include "ITStrackingGPU/Stream.h"
#include "ITStrackingGPU/Utils.h"
#include "GPUCommonLogger.h"

namespace o2
{
namespace its
{
namespace gpu
{
using utils::checkGPUError;

Stream::Stream()
{
  checkGPUError(cudaStreamCreate(&mStream));
}
Stream::~Stream()
{
  LOGP(info, "Destroying stream");
  checkGPUError(cudaStreamDestroy(mStream));
}

const GPUStream& Stream::get() const
{
  return mStream;
}

} // namespace gpu
} // namespace its
} // namespace o2