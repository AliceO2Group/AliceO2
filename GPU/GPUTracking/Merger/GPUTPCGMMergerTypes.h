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
#include "GPUGeneralKernels.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
namespace gputpcgmmergertypes
{

enum attachTypes { attachAttached = 0x40000000,
                   attachGood = 0x20000000,
                   attachGoodLeg = 0x10000000,
                   attachTube = 0x08000000,
                   attachHighIncl = 0x04000000,
                   attachTrackMask = 0x03FFFFFF,
                   attachFlagMask = 0xFC000000,
                   attachZero = 0 };

struct InterpolationErrorHit {
  float posY, posZ;
  GPUCA_MERGER_INTERPOLATION_ERROR_TYPE errorY, errorZ;
};

struct InterpolationErrors {
  InterpolationErrorHit hit[GPUCA_MERGER_MAX_TRACK_CLUSTERS];
};

struct GPUResolveSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<short, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCGMMergerResolve_step3)> {
  int iTrack1[GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCGMMergerResolve_step3)];
  int iTrack2[GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCGMMergerResolve_step3)];
};

struct GPUTPCGMBorderRange {
  int fId;
  float fMin, fMax;
};

} // namespace gputpcgmmergertypes
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
