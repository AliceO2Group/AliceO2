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
namespace gpu
{
template <int NLayers>
struct StaticTrackingParameters {
  // StaticTrackingParameters<NLayers>& operator=(const StaticTrackingParameters<NLayers>& t);
  // int CellMinimumLevel();
  /// General parameters
  int ClusterSharing = 0;
  int MinTrackLength = NLayers;
  /// Trackleting cuts
  float TrackletMaxDeltaPhi = 0.3f;
  float TrackletMaxDeltaZ[NLayers - 1] = {0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f};
  /// Cell finding cuts
  // float CellMaxDeltaTanLambda = 0.025f;
  // float CellMaxDCA[NLayers - 2] = {0.05f, 0.04f, 0.05f, 0.2f, 0.4f};
  // float CellMaxDeltaPhi = 0.14f;
  // float CellMaxDeltaZ[NLayers - 2] = {0.2f, 0.4f, 0.5f, 0.6f, 3.0f};
  // /// Neighbour finding cuts
  // float NeighbourMaxDeltaCurvature[NLayers - 3] = {0.008f, 0.0025f, 0.003f, 0.0035f};
  // float NeighbourMaxDeltaN[NLayers - 3] = {0.002f, 0.0090f, 0.002f, 0.005f};
};
} // namespace gpu

template <int NLayers = 7>
class TrackerTraitsGPU : public TrackerTraits
{
 public:
  TrackerTraitsGPU() = default;
  ~TrackerTraitsGPU() override = default;

  // void computeLayerCells() final;
  void adoptTimeFrame(TimeFrame* tf) override;
  void initialiseTimeFrame(const int iteration, const TrackingParameters& trackingParams) override;
  void computeLayerTracklets(const int iteration) final;
  void computeLayerCells(const int iteration) override;
  // void refitTracks(const std::vector<std::vector<TrackingFrameInfo>>& tf, std::vector<TrackITSExt>& tracks) override;

 private:
  gpu::TimeFrameGPU<7>* mTimeFrameGPU;
  gpu::StaticTrackingParameters<NLayers>* mStaticTrkPars;
};

template <int NLayers>
inline void TrackerTraitsGPU<NLayers>::adoptTimeFrame(TimeFrame* tf)
{
  mTimeFrameGPU = static_cast<gpu::TimeFrameGPU<NLayers>*>(tf);
}
} // namespace its
} // namespace o2

#endif