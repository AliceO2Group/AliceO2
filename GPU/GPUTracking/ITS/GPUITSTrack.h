// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUITSTrack.h
/// \author David Rohr, Maximiliano Puccio

#ifndef GPUITSTRACK_H
#define GPUITSTRACK_H

#include "GPUTPCGMMergerTypes.h"
#include "GPUTPCGMTrackParam.h"

namespace GPUCA_NAMESPACE::gpu
{
class GPUITSTrack : public GPUTPCGMTrackParam
{
 public:
  gputpcgmmergertypes::GPUTPCOuterParam mOuterParam;
  float mAlpha;
  int mClusters[7];
};
} // namespace GPUCA_NAMESPACE::gpu

#endif
