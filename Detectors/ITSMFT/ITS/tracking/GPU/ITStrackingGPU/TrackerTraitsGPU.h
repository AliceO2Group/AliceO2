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

#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2
{
namespace its
{

template <int nLayers = 7>
class TrackerTraitsGPU : public TrackerTraits
{
 public:
  TrackerTraitsGPU() = default;
  ~TrackerTraitsGPU() override = default;

  // void computeLayerCells() final;
  void adoptTimeFrame(TimeFrame* tf) override;
  void initialiseTimeFrame(const int iteration) override;
  void computeLayerTracklets(const int iteration) final;
  void computeLayerCells(const int iteration) override;
  void setBz(float) override;
  void findCellsNeighbours(const int iteration) override;
  void findRoads(const int iteration) override;

  // Methods to get CPU execution from traits
  void initialiseTimeFrameHybrid(const int iteration) override { initialiseTimeFrame(iteration); };
  void computeTrackletsHybrid(const int iteration) override;
  void computeCellsHybrid(const int iteration) override;
  void findCellsNeighboursHybrid(const int iteration) override;
  void findRoadsHybrid(const int iteration) override;
  void findTracksHybrid(const int iteration) override;

  void findTracks() override;
  void extendTracks(const int iteration) override;

  // TimeFrameGPU information forwarding
  int getTFNumberOfClusters() const override;
  int getTFNumberOfTracklets() const override;
  int getTFNumberOfCells() const override;

 private:
  IndexTableUtils* mDeviceIndexTableUtils;
  gpu::TimeFrameGPU<7>* mTimeFrameGPU;
  gpu::StaticTrackingParameters<nLayers>* mStaticTrkPars;
};

template <int nLayers>
inline void TrackerTraitsGPU<nLayers>::adoptTimeFrame(TimeFrame* tf)
{
  mTimeFrameGPU = static_cast<gpu::TimeFrameGPU<nLayers>*>(tf);
  mTimeFrame = static_cast<TimeFrame*>(tf);
}
} // namespace its
} // namespace o2

#endif