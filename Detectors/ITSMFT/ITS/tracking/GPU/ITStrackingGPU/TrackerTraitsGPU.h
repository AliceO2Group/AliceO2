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

#ifndef ITSTRACKINGGPU_TRACKERTRAITSGPU_H_
#define ITSTRACKINGGPU_TRACKERTRAITSGPU_H_

#include "ITStracking/TrackerTraits.h"
#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2::its
{

template <int nLayers = 7>
class TrackerTraitsGPU final : public TrackerTraits<nLayers>
{
  using typename TrackerTraits<nLayers>::IndexTableUtilsN;

 public:
  TrackerTraitsGPU() = default;
  ~TrackerTraitsGPU() final = default;

  void adoptTimeFrame(TimeFrame<nLayers>* tf) final;
  void initialiseTimeFrame(const int iteration) final;

  void computeLayerTracklets(const int iteration, int, int) final;
  void computeLayerCells(const int iteration) final;
  void findCellsNeighbours(const int iteration) final;
  void findRoads(const int iteration) final;

  bool supportsExtendTracks() const noexcept final { return false; }
  bool supportsFindShortPrimaries() const noexcept final { return false; }

  void setBz(float) final;

  const char* getName() const noexcept final { return "GPU"; }
  bool isGPU() const noexcept final { return true; }

  // TimeFrameGPU information forwarding
  int getTFNumberOfClusters() const override;
  int getTFNumberOfTracklets() const override;
  int getTFNumberOfCells() const override;

 private:
  IndexTableUtilsN* mDeviceIndexTableUtils;
  gpu::TimeFrameGPU<nLayers>* mTimeFrameGPU;
};

} // namespace o2::its

#endif
