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

/// \file GPUReconstructionHelpers.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONHELPERS_H
#define GPURECONSTRUCTIONHELPERS_H

#include <mutex>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUReconstructionDeviceBase;
class GPUReconstructionHelpers
{
 public:
  class helperDelegateBase
  {
  };

  struct helperParam {
    pthread_t threadId;
    GPUReconstructionDeviceBase* cls;
    int32_t num;
    std::mutex mutex[2];
    int8_t terminate;
    helperDelegateBase* functionCls;
    int32_t (helperDelegateBase::*function)(int32_t, int32_t, helperParam*);
    int32_t phase;
    int32_t count;
    volatile int32_t done;
    volatile int8_t error;
    volatile int8_t reset;
  };
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
