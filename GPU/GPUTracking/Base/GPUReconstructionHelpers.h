// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
    int num;
    std::mutex mutex[2];
    char terminate;
    helperDelegateBase* functionCls;
    int (helperDelegateBase::*function)(int, int, helperParam*);
    int phase;
    int count;
    volatile int done;
    volatile char error;
    volatile char reset;
  };
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
