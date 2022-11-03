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
#include "CommonUtils/StringUtils.h"
#include "Framework/DataTakingContext.h"
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

  void init(const TrackResiduals* residualsEngine, std::string outputDir, bool wFile, bool wBinnedResid, bool wUnbinnedResid, bool wTrackData, int autosave, int compression);
  void fillStatisticsBranches();
  uint64_t getNEntries() const { return nResidualsTotal; }

  void fill(const o2::dataformats::TFIDInfo& ti, const std::pair<gsl::span<const o2::tpc::TrackData>, gsl::span<const UnbinnedResid>> data);
  void merge(ResidualsContainer* prev);
  void print();
  void writeToFile(bool closeFileAfterwards);

  const TrackResiduals* trackResiduals{nullptr};
  std::array<std::vector<TrackResiduals::LocalResid>, SECTORSPERSIDE * SIDES> residuals{}; ///< local residuals per sector which are sent to the aggregator
  std::array<std::vector<TrackResiduals::LocalResid>*, SECTORSPERSIDE * SIDES> residualsPtr{};
  std::array<std::vector<TrackResiduals::VoxStats>, SECTORSPERSIDE * SIDES> stats{}; ///< voxel statistics sent to the aggregator
  std::array<std::vector<TrackResiduals::VoxStats>*, SECTORSPERSIDE * SIDES> statsPtr{};
  uint32_t runNumber;                                                        ///< run number (required for meta data file)
  std::vector<uint32_t> tfOrbits, *tfOrbitsPtr{&tfOrbits};                   ///< first TF orbit
  std::vector<uint32_t> sumOfResiduals, *sumOfResidualsPtr{&sumOfResiduals}; ///< sum of residuals for each TF
  std::vector<UnbinnedResid> unbinnedRes, *unbinnedResPtr{&unbinnedRes};     // unbinned residuals
  std::vector<TrackData> trkData, *trkDataPtr{&trkData};                                 // track data and cluster ranges

  std::string fileName{"o2tpc_residuals"};
  std::string treeNameResiduals{"resid"};
  std::string treeNameStats{"stats"};
  std::string treeNameRecords{"records"};
  std::unique_ptr<TFile> fileOut{nullptr};
  std::unique_ptr<TTree> treeOutResidualsUnbinned{nullptr};
  std::unique_ptr<TTree> treeOutTrackData{nullptr};
  std::unique_ptr<TTree> treeOutResiduals{nullptr};
  std::unique_ptr<TTree> treeOutStats{nullptr};
  std::unique_ptr<TTree> treeOutRecords{nullptr};

  bool writeToRootFile{true};
  bool writeBinnedResid{false};
  bool writeUnbinnedResiduals{false};
  bool writeTrackData{false};
  int autosaveInterval{0};

  uint64_t nResidualsTotal{0};

  ClassDefNV(ResidualsContainer, 3);
};

class ResidualAggregator final : public o2::calibration::TimeSlotCalibration<UnbinnedResid, ResidualsContainer>
{
  using Slot = o2::calibration::TimeSlot<ResidualsContainer>;

 public:
  ResidualAggregator(size_t nMin = 1000) : mMinEntries(nMin) { mTrackResiduals.init(); }
  ~ResidualAggregator() final;

  void setDataTakingContext(o2::framework::DataTakingContext& dtc) { mDataTakingContext = dtc; }
  void setOutputDir(std::string dir) { mOutputDir = dir.empty() ? o2::utils::Str::rectifyDirectory("./") : dir; }
  void setMetaFileOutputDir(std::string dir)
  {
    mMetaOutputDir = dir;
    mStoreMetaData = true;
  }
  void setLHCPeriod(std::string period) { mLHCPeriod = period; }
  void setWriteBinnedResiduals(bool f) { mWriteBinnedResiduals = f; }
  void setWriteUnbinnedResiduals(bool f) { mWriteUnbinnedResiduals = f; }
  void setWriteTrackData(bool f) { mWriteTrackData = f; }
  void setAutosaveInterval(int n) { mAutosaveInterval = n; }
  void disableFileWriting() { mWriteOutput = false; }
  void setCompression(int c) { mCompressionSetting = c; }

  bool hasEnoughData(const Slot& slot) const final;
  void initOutput() final;
  void finalizeSlot(Slot& slot) final;
  Slot& emplaceNewSlot(bool front, TFType tStart, TFType tEnd) final;

 private:
  o2::framework::DataTakingContext mDataTakingContext{};
  TrackResiduals mTrackResiduals; ///< providing the functionality for voxel binning of the residuals
  std::string mOutputDir{"./"};   ///< the directory where the output of residuals is stored
  std::string mMetaOutputDir{"none"}; ///< the directory where the meta data file is stored
  std::string mLHCPeriod{""};         ///< the LHC period to be put into the meta file
  bool mStoreMetaData{false};         ///< flag, whether meta file is supposed to be stored
  bool mWriteOutput{true};            ///< if false, no output files will be written
  bool mWriteBinnedResiduals{false};  ///< flag, whether to write binned residuals to output file
  bool mWriteUnbinnedResiduals{false}; ///< flag, whether to write unbinned residuals to output file
  bool mWriteTrackData{false};         ///< flag, whether to write track data to output file
  int mAutosaveInterval{0};            ///< if >0 then the output is written to a file for every n-th TF
  int mCompressionSetting{101};        ///< single integer defining the ROOT compression algorithm and level (see TFile doc for details)
  size_t mMinEntries;             ///< the minimum number of residuals required for the map creation (per voxel)

  ClassDefOverride(ResidualAggregator, 4);
};

} // namespace tpc
} // namespace o2

#endif // O2_TPC_RESIDUALAGGREGATOR_H
