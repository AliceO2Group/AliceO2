// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMMergerTypes.h
/// \author David Rohr

#ifndef GPUTPCGMMERGERTYPES_H
#define GPUTPCGMMERGERTYPES_H

#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
namespace GPUTPCGMMergerTypes
{

enum attachTypes { attachAttached = 0x40000000,
                   attachGood = 0x20000000,
                   attachGoodLeg = 0x10000000,
                   attachTube = 0x08000000,
                   attachHighIncl = 0x04000000,
                   attachTrackMask = 0x03FFFFFF,
                   attachFlagMask = 0xFC000000 };

struct InterpolationErrorHit {
  float posY;
  float errorY;
  float posZ;
  float errorZ;
};

struct InterpolationErrors {
  InterpolationErrorHit hit[GPUCA_MERGER_MAX_TRACK_CLUSTERS];
};

} // namespace GPUTPCGMMergerTypes
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
