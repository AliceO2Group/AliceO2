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
/// \file TrackerTraitsNV.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKERTRAITSNV_H_
#define TRACKINGITSU_INCLUDE_TRACKERTRAITSNV_H_

#ifndef GPUCA_GPUCODE_GENRTC
#include <cub/cub.cuh>
#include <cstdint>
#endif
#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/TrackerTraits.h"

namespace o2
{
namespace its
{

class PrimaryVertexContext;

class TrackerTraitsNV : public TrackerTraits
{
 public:
  TrackerTraitsNV() = default;
  ~TrackerTraitsNV() override = default;

  void computeLayerCells() final;
  void computeLayerTracklets() final;
  void refitTracks(const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks) override;
};

extern "C" TrackerTraits* createTrackerTraitsNV();
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
