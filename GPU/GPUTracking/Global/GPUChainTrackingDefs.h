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

/// \file GPUChainTracking.h
/// \author David Rohr

#ifndef GPUCHAINTRACKINGDEFS_H
#define GPUCHAINTRACKINGDEFS_H

#include <mutex>
#include <condition_variable>

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUChainTrackingFinalContext {
  GPUReconstruction* rec = nullptr;
  std::mutex mutex;
  std::condition_variable cond;
  bool ready = false;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
