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
#include "DataFormatsCTP/LumiInfo.h"
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

  using TFType = o2::calibration::TFType;

  ResidualsContainer() = default;
  ResidualsContainer(ResidualsContainer&& rhs);
  ResidualsContainer(const ResidualsContainer&); // no copying allowed, this will yield an error message
  ResidualsContainer& operator=(const ResidualsContainer& src) = delete;
  ~ResidualsContainer();

  void init(const TrackResiduals* residualsEngine, std::string outputDir, bool wFile, bool wBinnedResid, bool wUnbinnedResid, bool wTrackData, int autosave, int compression, long orbitResetTime);
  void fillStatisticsBranches();
  uint64_t getNEntries() const { return nResidualsTotal; }

  void fill(const o2::dataformats::TFIDInfo& ti, const gsl::span<const UnbinnedResid> resid, const gsl::span<const o2::tpc::TrackDataCompact> trkRefsIn, const gsl::span<const o2::tpc::TrackData>* trkDataIn, const o2::ctp::LumiInfo* lumiInput);
  void merge(ResidualsContainer* prev);
  void print();
  void writeToFile(bool closeFileAfterwards);

  const TrackResiduals* trackResiduals{nullptr};
  std::array<std::vector<TrackResiduals::LocalResid>, SECTORSPERSIDE * SIDES> residuals{}; ///< local (binned) residuals per sector
  std::array<std::vector<TrackResiduals::LocalResid>*, SECTORSPERSIDE * SIDES> residualsPtr{};
  std::array<std::vector<TrackResiduals::VoxStats>, SECTORSPERSIDE * SIDES> stats{}; ///< voxel statistics per sector
  std::array<std::vector<TrackResiduals::VoxStats>*, SECTORSPERSIDE * SIDES> statsPtr{};
  std::vector<uint32_t> tfOrbits, *tfOrbitsPtr{&tfOrbits};                   ///< first TF orbit
  std::vector<uint32_t> sumBinnedResid, *sumBinnedResidPtr{&sumBinnedResid}; ///< sum of binned residuals for each TF
  std::vector<uint32_t> sumUnbinnedResid, *sumUnbinnedResidPtr{&sumUnbinnedResid}; ///< sum of unbinned residuals for each TF
  std::vector<o2::ctp::LumiInfo> lumi, *lumiPtr{&lumi};                      ///< luminosity information from CTP per TF
  std::vector<UnbinnedResid> unbinnedRes, *unbinnedResPtr{&unbinnedRes};     ///< unbinned residuals which are sent to the aggregator
  std::vector<TrackData> trkData, *trkDataPtr{&trkData};                     ///< track data and cluster ranges
  std::vector<TrackDataCompact> trackInfo, *trackInfoPtr{&trackInfo};        ///< allows to obtain track type for each unbinned residual downstream
  o2::ctp::LumiInfo lumiTF;                                                  ///< for each processed TF we store the lumi information in the tree of unbinned residuals
  uint64_t timeMS;                                                           ///< for each processed TF we store its absolute time in ms in the tree of unbinned residuals

  std::string fileName{"o2tpc_residuals"};
  std::unique_ptr<TFile> fileOut{nullptr};
  std::unique_ptr<TTree> treeOutResidualsUnbinned{nullptr};
  std::unique_ptr<TTree> treeOutTrackData{nullptr};
  std::unique_ptr<TTree> treeOutResiduals{nullptr};
  std::unique_ptr<TTree> treeOutStats{nullptr};
  std::unique_ptr<TTree> treeOutRecords{nullptr};

  // settings
  bool writeToRootFile{true};         ///< set to false to avoid that any output file is produced
  bool writeBinnedResid{false};       ///< flag, whether binned residuals should be written out
  bool writeUnbinnedResiduals{false}; ///< flag, whether unbinned residuals should be written out
  bool writeTrackData{false};         ///< flag, whether full seeding track information should be written out
  int autosaveInterval{0};            ///< if > 0, then the output written to file for every n-th TF

  // additional info
  long orbitReset{0};                               ///< current orbit reset time in ms
  uint32_t firstTForbit{0};                         ///< stored for the first seen TF to allow conversion to time stamp
  TFType firstSeenTF{o2::calibration::INFINITE_TF}; ///< the first TF which was added to this container
  TFType lastSeenTF{0};                             ///< the last TF which was added to this container
  uint64_t nResidualsTotal{0};                      ///< the total number of residuals for this container
  float TPCVDriftRef{-1.};                          ///< TPC nominal drift speed in cm/microseconds
  float TPCDriftTimeOffsetRef{0.};                  ///< TPC nominal (e.g. at the start of run) drift time bias in cm/mus

  ClassDefNV(ResidualsContainer, 5);
};

class ResidualAggregator final : public o2::calibration::TimeSlotCalibration<ResidualsContainer>
{
  using Slot = o2::calibration::TimeSlot<ResidualsContainer>;

 public:
  ResidualAggregator(size_t nMin = 1000) : mMinEntries(nMin) { mTrackResiduals.init(); }
  ~ResidualAggregator() final;

  void setDataTakingContext(o2::framework::DataTakingContext& dtc) { mDataTakingContext = dtc; }
  void setOrbitResetTime(long t) { mOrbitResetTime = t; }
  void setTPCVDrift(const o2::tpc::VDriftCorrFact& v);
  void setOutputDir(std::string dir)
  {
    mOutputDir = dir;
    mWriteOutput = true;
  }
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
  bool mWriteOutput{false};           ///< flag, whether output files are supposed to be stored
  bool mWriteBinnedResiduals{false};  ///< flag, whether to write binned residuals to output file
  bool mWriteUnbinnedResiduals{false}; ///< flag, whether to write unbinned residuals to output file
  bool mWriteTrackData{false};         ///< flag, whether to write track data to output file
  int mAutosaveInterval{0};            ///< if >0 then the output is written to a file for every n-th TF
  int mCompressionSetting{101};        ///< single integer defining the ROOT compression algorithm and level (see TFile doc for details)
  size_t mMinEntries;             ///< the minimum number of residuals required for the map creation (per voxel)
  long mOrbitResetTime;           ///< orbit reset time in ms
  float mTPCVDriftRef = -1.;      ///< TPC nominal drift speed in cm/microseconds
  float mTPCDriftTimeOffsetRef = 0.; ///< TPC nominal (e.g. at the start of run) drift time bias in cm/mus

  ClassDefOverride(ResidualAggregator, 4);
};

} // namespace tpc
} // namespace o2

#endif // O2_TPC_RESIDUALAGGREGATOR_H
