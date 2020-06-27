// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file TrackerTraits.h
/// \brief
///

#ifndef TRACKINGEC0__INCLUDE_TRACKERTRAITSCPU_H_
#define TRACKINGEC0__INCLUDE_TRACKERTRAITSCPU_H_

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>

#include "EC0tracking/TrackerTraits.h"
#include "EC0tracking/Configuration.h"
#include "EC0tracking/Definitions.h"
#include "EC0tracking/MathUtils.h"
#include "EC0tracking/PrimaryVertexContext.h"
#include "EC0tracking/Road.h"

namespace o2
{
namespace ecl
{

class TrackerTraitsCPU : public TrackerTraits
{
 public:
  TrackerTraitsCPU() { mPrimaryVertexContext = new PrimaryVertexContext; }
  ~TrackerTraitsCPU() override { delete mPrimaryVertexContext; }

  void computeLayerCells() final;
  void computeLayerTracklets() final;
  void refitTracks(const std::array<std::vector<TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITSExt>& tracks) final;

 protected:
  std::vector<std::vector<Tracklet>> mTracklets;
  std::vector<std::vector<Cell>> mCells;
};
} // namespace ecl
} // namespace o2

#endif /* TRACKINGEC0__INCLUDE_TRACKERTRAITS_H_ */
