// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMMergedTrackHit.h
/// \author David Rohr

#ifndef GPUTPCGMMERGEDTRACKHIT_H
#define GPUTPCGMMERGEDTRACKHIT_H

#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCGMMergedTrackHit {
#ifndef LATE_TPC_TRANSFORM
  float x, y, z;
#endif
  unsigned int num;
  unsigned char slice, row, leg, state;
#ifndef LATE_TPC_TRANSFORM
  unsigned short amp;
#endif

  enum hitState { flagSplitPad = 0x1,
                  flagSplitTime = 0x2,
                  flagSplit = 0x3,
                  flagEdge = 0x4,
                  flagSingle = 0x8,
                  flagShared = 0x10,
                  hwcmFlags = 0x1F,
                  flagRejectDistance = 0x20,
                  flagRejectErr = 0x40,
                  flagReject = 0x60,
                  flagNotFit = 0x80 };

#ifdef GPUCA_TPC_RAW_PROPAGATE_PAD_ROW_TIME
 public:
  float pad;
  float time;
#endif
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
