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

#ifndef ITSTRACKINGGPU_TRACKINGKERNELS_H_
#define ITSTRACKINGGPU_TRACKINGKERNELS_H_

#include "DetectorsBase/Propagator.h"
#include "GPUCommonDef.h"

namespace o2::its
{
class CellSeed;
namespace gpu
{
#ifdef GPUCA_GPUCODE // GPUg() global kernels must only when compiled by GPU compiler
GPUd() bool fitTrack(TrackITSExt& track,
                     int start,
                     int end,
                     int step,
                     float chi2clcut,
                     float chi2ndfcut,
                     float maxQoverPt,
                     int nCl,
                     float Bz,
                     TrackingFrameInfo** tfInfos,
                     const o2::base::Propagator* prop,
                     o2::base::PropagatorF::MatCorrType matCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE);

template <int nLayers = 7>
GPUg() void fitTrackSeedsKernel(
  CellSeed* trackSeeds,
  TrackingFrameInfo** foundTrackingFrameInfo,
  o2::its::TrackITSExt* tracks,
  const unsigned int nSeeds,
  const float Bz,
  const int startLevel,
  float maxChi2ClusterAttachment,
  float maxChi2NDF,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType = o2::base::PropagatorF::MatCorrType::USEMatCorrLUT);
#endif
} // namespace gpu

void trackSeedHandler(CellSeed* trackSeeds,
                      TrackingFrameInfo** foundTrackingFrameInfo,
                      o2::its::TrackITSExt* tracks,
                      const unsigned int nSeeds,
                      const float Bz,
                      const int startLevel,
                      float maxChi2ClusterAttachment,
                      float maxChi2NDF,
                      const o2::base::Propagator* propagator,
                      const o2::base::PropagatorF::MatCorrType matCorrType);
} // namespace o2::its
#endif // ITSTRACKINGGPU_TRACKINGKERNELS_H_
