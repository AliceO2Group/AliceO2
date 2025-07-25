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

/// \file GPUTPCGMMergedTrackHit.h
/// \author David Rohr

#ifndef GPUTPCGMMERGEDTRACKHIT_H
#define GPUTPCGMMERGEDTRACKHIT_H

#include "GPUCommonDef.h"

namespace o2::gpu
{
struct GPUTPCGMMergedTrackHit {
  uint32_t num;
  uint8_t sector, row, state;

  // NOTE: the lower states must match those from ClusterNative!
  // TODO: take them directly from clusterNative header.
  enum hitState { flagSplitPad = 0x1,
                  flagSplitTime = 0x2,
                  flagSplit = 0x3,
                  flagEdge = 0x4,
                  flagSingle = 0x8,
                  flagShared = 0x10,
                  clustererAndSharedFlags = 0x1F,
                  flagRejectDistance = 0x20,
                  flagRejectErr = 0x40,
                  flagReject = 0x60,
                  flagHighIncl = 0x80 };
};

} // namespace o2::gpu

#endif
