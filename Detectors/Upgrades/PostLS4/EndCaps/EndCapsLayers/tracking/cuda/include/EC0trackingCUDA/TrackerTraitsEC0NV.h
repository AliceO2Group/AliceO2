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
/// \file TrackerTraitsEC0NV.h
/// \brief
///

#ifndef TRACKINGEC0__INCLUDE_TRACKERTRAITSNV_H_
#define TRACKINGEC0__INCLUDE_TRACKERTRAITSNV_H_

#include "EC0tracking/Configuration.h"
#include "EC0tracking/Definitions.h"
#include "EC0tracking/TrackerTraitsEC0.h"

namespace o2
{
namespace ecl
{

class PrimaryVertexContext;

class TrackerTraitsEC0NV : public TrackerTraitsEC0
{
 public:
  TrackerTraitsEC0NV();
  virtual ~TrackerTraitsEC0NV();

  void computeLayerCells() final;
  void computeLayerTracklets() final;
  void refitTracks(const std::array<std::vector<TrackingFrameInfo>, 7>& tf, std::vector<o2::its::TrackITSExt>& tracks) final;
};

extern "C" TrackerTraitsEC0* createTrackerTraitsEC0NV();
} // namespace ecl
} // namespace o2

#endif /* TRACKINGEC0__INCLUDE_TRACKERTRAITS_H_ */
