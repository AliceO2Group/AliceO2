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

namespace o2
{
namespace gpu
{
struct GPUTPCGMMergedTrackHit {
  float x, y, z;
  unsigned int num;
  unsigned char slice, row, leg, state;
  unsigned short amp;

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

#ifdef GMPropagatePadRowTime
 public:
  float pad;
  float time;
#endif
};
}
} // namespace o2::gpu

#endif
