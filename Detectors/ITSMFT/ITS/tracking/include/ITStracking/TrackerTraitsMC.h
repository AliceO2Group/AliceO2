// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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

#ifndef TRACKINGITSU_INCLUDE_TRACKERTRAITSMC_H_
#define TRACKINGITSU_INCLUDE_TRACKERTRAITSMC_H_

#include "ITStracking/TrackerTraits.h"

namespace o2::its
{

// This class faciliates the tuning of the reconstruction parameters.
// Two modes are foreseen:
// 1. Inspect and dump the artefacts the reconstrcution produces
// 2. Run the same over MC truth
// Both should faciliate finding sources of in-efficiency (c.f., missing links)
// and allow to tune overall the imposed parameters in a consistent way.
template <int NLayers>
class TrackerTraitsMC : public TrackerTraits<NLayers>
{
 public:
  TrackerTraitsMC() = default;
  ~TrackerTraitsMC() override = default;

  void computeLayerTracklets(const int iteration, int iVertex) override;
  void computeLayerCells(const int iteration) override;
  void findCellsNeighbours(const int iteration) override;
  void findRoads(const int iteration) override;

  const char* getName() const noexcept override { return "TUNE"; }
};

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITSMC_H_ */
