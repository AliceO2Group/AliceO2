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

#ifndef ALICEO2_ITSMFT_TRACKING_TEST_TRAVERSALTESTSUPPORT_H_
#define ALICEO2_ITSMFT_TRACKING_TEST_TRAVERSALTESTSUPPORT_H_

#include <stdexcept>

#include "ITSMFTTracking/Tracker.h"

namespace o2::itsmft::tracking
{

// Test-only access to the Tracker-owned initialization transaction and
// explicit backend stages. The caller owns the span buffer for the returned view.
struct TrackerTestAccess {
  static IterationContext prepare(Tracker& tracker, TimeFrame& frame, int iteration,
                                  std::array<gsl::span<const GlobalMeasurement>, MaxLayoutSurfaces>& measurementSpans)
  {
    const auto* configuration = iteration < 0 ? nullptr : tracker.getIterationConfiguration(static_cast<std::size_t>(iteration));
    if (configuration == nullptr || !tracker.isConfiguredFor(frame)) {
      throw std::out_of_range{"test traversal iteration"};
    }
    auto& scratch = frame.getScratch();
    auto layerGlobalMeasurements = tracker.prepareTimeFrame(frame, measurementSpans);
    IterationContext view{iteration,
                          frame,
                          scratch,
                          configuration->getTopologyView(frame.getLayout().getSurfaceCatalog()),
                          *configuration,
                          tracker.mDetectorConfiguration,
                          layerGlobalMeasurements,
                          frame.getBz()};
    tracker.initializeIteration(view);
    return view;
  }

  static void computeTracklets(TrackerTraits& traits, IterationContext& view, int vertex)
  {
    traits.computeLayerTracklets(view, view.iteration, vertex);
  }

  static void computeCells(TrackerTraits& traits, IterationContext& view)
  {
    traits.computeLayerCells(view, view.iteration);
  }

  static void findNeighbours(TrackerTraits& traits, IterationContext& view)
  {
    traits.findCellsNeighbours(view, view.iteration);
  }

  static bool buildTrackSeed(TrackerTraits& traits, IterationContext& view,
                             int cellPathId, const CellSeed& cell,
                             TrackSeed& output, OperationFailureReason& reason)
  {
    return traits.buildTrackSeed(view, cellPathId, cell, output, reason);
  }

  static void findRoads(TrackerTraits& traits, IterationContext& view)
  {
    traits.findRoads(view, view.iteration);
  }

  static void computeTracksMClabels(Tracker& tracker, TimeFrame& frame)
  {
    tracker.computeTracksMClabels(frame);
  }

  static void configureBeamPosition(Tracker& tracker, TimeFrame& frame)
  {
    tracker.configureBeamPosition(frame);
  }
};

} // namespace o2::itsmft::tracking

#endif
