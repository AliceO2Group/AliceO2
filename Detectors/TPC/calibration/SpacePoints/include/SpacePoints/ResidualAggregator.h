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

/// \file ResidualAggregator.h
/// \brief Collects local TPC cluster residuals from EPNs
/// \author Ole Schmidt

#ifndef O2_TPC_RESIDUALAGGREGATOR_H
#define O2_TPC_RESIDUALAGGREGATOR_H

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTPC/Defs.h"
#include "SpacePoints/TrackResiduals.h"

#include <vector>
#include <array>
#include <string>

#include "TFile.h"
#include "TTree.h"

namespace o2
{
namespace tpc
{

struct ResidualsContainer {

  ResidualsContainer() = default;
  ResidualsContainer(ResidualsContainer&& rhs);
  ResidualsContainer(const ResidualsContainer&); // no copying allowed, this will yield an error message
  ResidualsContainer& operator=(const ResidualsContainer& src) = delete;
  ~ResidualsContainer();

  void init(const TrackResiduals* residualsEngine);
  void fillStatisticsBranches();
  uint64_t getNEntries() const { return nResidualsTotal; }

  void fill(const gsl::span<const TrackResiduals::UnbinnedResid> data);
  void merge(ResidualsContainer* prev);
  void print();

  const TrackResiduals* trackResiduals{nullptr};
  std::array<std::vector<TrackResiduals::LocalResid>, SECTORSPERSIDE * SIDES> residuals{}; ///< local residuals per sector which are sent to the aggregator
  std::array<std::vector<TrackResiduals::LocalResid>*, SECTORSPERSIDE * SIDES> residualsPtr{};
  std::array<std::vector<TrackResiduals::VoxStats>, SECTORSPERSIDE * SIDES> stats{}; ///< voxel statistics sent to the aggregator
  std::array<std::vector<TrackResiduals::VoxStats>*, SECTORSPERSIDE * SIDES> statsPtr{};

  std::string fileName{"o2tpc_residuals"};
  std::string treeNameResiduals{"resid"};
  std::string treeNameStats{"stats"};
  std::unique_ptr<TFile> fileOut{nullptr};
  std::unique_ptr<TTree> treeOutResiduals{nullptr};
  std::unique_ptr<TTree> treeOutStats{nullptr};

  uint64_t nResidualsTotal{0};

  ClassDefNV(ResidualsContainer, 1);
};

class ResidualAggregator final : public o2::calibration::TimeSlotCalibration<TrackResiduals::UnbinnedResid, ResidualsContainer>
{
  using Slot = o2::calibration::TimeSlot<ResidualsContainer>;

 public:
  ResidualAggregator(size_t nMin = 1'000) : mMinEntries(nMin) { mTrackResiduals.init(); }
  ~ResidualAggregator() final;

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, uint64_t tStart, uint64_t tEnd) final;

 private:
  TrackResiduals mTrackResiduals; ///< providing the functionality for voxel binning of the residuals
  size_t mMinEntries;             ///< the minimum number of residuals required for the map creation (per voxel)

  ClassDefOverride(ResidualAggregator, 1);
};

} // namespace tpc
} // namespace o2

#endif // O2_TPC_RESIDUALAGGREGATOR_H
