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

/// \file calculatedEdx.C
/// \brief Example macro showing how to use o2::tpc::CalculatedEdx to calculate TPC dE/dx from TPC tracks and native clusters.
///        Supports real data (CTF- or TF-reconstructed) and MC productions, and optionally restricts the calculation to TPC tracks matched to an ITS track.
///        Accepts a list of dEdx settings (truncation range, correction mask, cluster mask, subthreshold/stack-boundary method).
///        The track refit/propagation is performed only once per track no matter how many settings entries are given.
///        The output tree has one row per track, with one "dEdx<i>" branch per entry in settingsList (see the "dEdx<i>: low=..., high=..." log lines for what each index means).

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include "TFile.h"
#include "TROOT.h"
#include "TTree.h"
#include "Framework/Logger.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonDataFormat/TFIDInfo.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TPCCalibration/CalculatedEdx.h"
#endif

using namespace o2::tpc;
namespace fs = std::filesystem;

namespace
{
TFile* openOrNull(const std::string& fileName)
{
  auto f = TFile::Open(fileName.data());
  if (!f || !f->IsOpen() || f->IsZombie()) {
    LOGP(error, "Could not open file {}", fileName);
    return nullptr;
  }
  return f;
}

/// open fileName and retrieve treeName from it; logs why and returns {nullptr, nullptr} if either step fails
std::pair<std::unique_ptr<TFile>, std::unique_ptr<TTree>> openTreeOrNull(const std::string& fileName, const char* treeName)
{
  std::unique_ptr<TFile> file(openOrNull(fileName));
  if (!file) {
    return {nullptr, nullptr};
  }
  std::unique_ptr<TTree> tree((TTree*)file->Get(treeName));
  if (!tree) {
    LOGP(error, "Could not find tree '{}' in {}", treeName, fileName);
    return {nullptr, nullptr};
  }
  return {std::move(file), std::move(tree)};
}

/// default single-entry settings list
std::vector<dEdxSettings> defaultSettingsList()
{
  dEdxSettings s;
  s.low = 0.015f;
  s.high = 0.6f;
  s.correctionMask = CorrectionFlags::TopologyPol | CorrectionFlags::dEdxResidual;
  s.clusterMask = ClusterFlags::ExcludeEdgeCl;
  s.subthresholdMethod = 0;
  s.stackBoundaryMethod = 0;
  s.sameRowClusterMethod = 2;
  return {s};
}
} // namespace

/// \param dir directory containing tpctracks.root, tpc-native-clusters.root and, if isMatchedToITS, o2trac_its.root/o2match_itstpc.root
/// \param runNumberOrTimeStamp run number or timestamp used to load the calibration objects from CCDB for every timeframe, overridden per timeframe whenever a tfIDFileName file is found in dir
/// \param outFile name of the output file with the calculated dE/dx tree
/// \param localCCDBFolder if non-empty, load calibration objects from local CCDB snapshots in this folder instead of from the CCDB server
/// \param settingsList list of dEdx settings to evaluate for every track (truncation range, correction/cluster mask,
///                     subthreshold/stack-boundary/same-row-cluster method)
/// \param useRefit refit the tracks at each cluster row using the GPU refitter (default)
/// \param propagateTrack propagate the full track, including material corrections, instead of refitting; only used if useRefit is false
/// \param propagateParams propagate only the track parameters (fastest option, no material corrections); only used if useRefit and propagateTrack are both false
/// \param debug enable the CalculatedEdx debug streamer, additionally writing one dEdxDebug.root file with per-cluster information (or dEdxDebug_s<j>.root per settings entry j, if settingsList has more than one entry)
/// \param isMC set to true for MC productions to read the true MC track label (TPCTracksMCTruth branch) for each track into an additional "mcLabel" branch
/// \param isMatchedToITS set to true to also read o2trac_its.root/o2match_itstpc.root and restrict the dE/dx calculation to TPC tracks matched to an ITS track
/// \param tfIDFileName name of the optional timeframe ID file; if found in dir, its per-entry time stamp is used for the CCDB instead of runNumberOrTimeStamp
/// \param tpcTracksFileName name of the file with the TPC tracks
/// \param clusterNativeFileName name of the file with the TPC native clusters
/// \param itsTracksFileName name of the file with the ITS tracks; only used if isMatchedToITS is true
/// \param matchFileName name of the file with the TPC-ITS match information; only used if isMatchedToITS is true
/// \param maxEvents if >= 0, process at most this many events, instead of all available ones, default -1 processes all events
/// \param applySCToRefitTransform put the space-charge-corrected cluster map into the refit transform; only used if useRefit is true
/// \param applyVDriftToRefitTransform apply the calibrated drift velocity/t0 to the refit transform; only used if useRefit is true
void calculatedEdx(const std::string dir = ".",
                   const long runNumberOrTimeStamp = 0,
                   const std::string outFile = "dEdxCalc.root",
                   const std::string localCCDBFolder = "",
                   const std::vector<dEdxSettings> settingsList = defaultSettingsList(),
                   const bool useRefit = true,
                   const bool propagateTrack = false,
                   const bool propagateParams = false,
                   const bool debug = false,
                   const bool isMC = false,
                   const bool isMatchedToITS = false,
                   const std::string tfIDFileName = "o2_tfidinfo.root",
                   const std::string tpcTracksFileName = "tpctracks.root",
                   const std::string clusterNativeFileName = "tpc-native-clusters.root",
                   const std::string itsTracksFileName = "o2trac_its.root",
                   const std::string matchFileName = "o2match_itstpc.root",
                   const long long maxEvents = -1,
                   const bool applySCToRefitTransform = false,
                   const bool applyVDriftToRefitTransform = true)
{
  if (settingsList.empty()) {
    LOGP(error, "settingsList must not be empty");
    return;
  }
  for (size_t i = 0; i < settingsList.size(); i++) {
    const auto& s = settingsList[i];
    if (s.low < 0.f || s.high > 1.f || s.low >= s.high) {
      LOGP(error, "settingsList[{}]: invalid truncation range [{}, {}); expected 0 <= low < high <= 1", i, s.low, s.high);
      return;
    }
    if (s.subthresholdMethod != 0 && s.subthresholdMethod != 1) {
      LOGP(error, "settingsList[{}]: invalid subthresholdMethod {}; expected 0 (minimum charge) or 1 (minimum charge / 2)", i, s.subthresholdMethod);
      return;
    }
    if (s.stackBoundaryMethod > 2) {
      LOGP(error, "settingsList[{}]: invalid stackBoundaryMethod {}; expected 0 (disabled), 1 (exclude boundary row) or 2 (also exclude the adjacent row)", i, s.stackBoundaryMethod);
      return;
    }
    if (s.sameRowClusterMethod > 2) {
      LOGP(error, "settingsList[{}]: invalid sameRowClusterMethod {}; expected 0 (do not merge), 1 (merge, sum qTot) or 2 (merge, highest qTot)", i, s.sameRowClusterMethod);
      return;
    }
    // the output tree stores one "dEdx<i>" branch per settingsList entry; log what each index means here
    LOGP(info, "dEdx{}: low={}, high={}, correctionMask={}, clusterMask={}, subthresholdMethod={}, stackBoundaryMethod={}, sameRowClusterMethod={}", i, s.low, s.high, static_cast<unsigned short>(s.correctionMask), static_cast<unsigned short>(s.clusterMask), s.subthresholdMethod, s.stackBoundaryMethod, s.sameRowClusterMethod);
  }

  const std::clock_t c_start = std::clock();
  const auto t_start = std::chrono::high_resolution_clock::now();

  CalculatedEdx calcdEdx;
  calcdEdx.setDebug(debug);
  calcdEdx.setPropagateTrack(propagateTrack);
  calcdEdx.setPropagateParams(propagateParams);

  // own copy of settingsList, with debugRootFile made unique per settings entry (if there is more than one) so debug streams from different settings never collide
  std::vector<dEdxSettings> activeSettingsList = settingsList;
  if (debug) {
    for (size_t iSettings = 0; iSettings < settingsList.size(); iSettings++) {
      activeSettingsList[iSettings].debugRootFile = (settingsList.size() == 1) ? "dEdxDebug.root" : fmt::format("dEdxDebug_s{}.root", iSettings);
    }
  }

  auto [tpcFile, tpcTree] = openTreeOrNull(fmt::format("{}/{}", dir, tpcTracksFileName), "tpcrec");
  if (!tpcTree) {
    return;
  }

  std::vector<o2::tpc::TrackTPC> tpcTracks, *tpcTracksPtr = &tpcTracks;
  std::vector<o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput{nullptr};
  tpcTree->SetBranchAddress("TPCTracks", &tpcTracksPtr);
  tpcTree->SetBranchAddress("ClusRefs", &tpcTrackClIdxVecInput);

  std::vector<o2::MCCompLabel> tpcMCTruth, *tpcMCTruthPtr = &tpcMCTruth;
  if (isMC) {
    if (!tpcTree->GetBranch("TPCTracksMCTruth")) {
      LOGP(error, "Branch 'TPCTracksMCTruth' not found in {}/{}, cannot resolve MC truth", dir, tpcTracksFileName);
      return;
    }
    tpcTree->SetBranchAddress("TPCTracksMCTruth", &tpcMCTruthPtr);
  }

  std::unique_ptr<TFile> itsFile;
  std::unique_ptr<TTree> itsTree;
  std::unique_ptr<TFile> matchFile;
  std::unique_ptr<TTree> matchTree;
  std::vector<o2::its::TrackITS> itsTracks, *itsTracksPtr = &itsTracks;
  std::vector<o2::dataformats::TrackTPCITS> matchTracks, *matchTracksPtr = &matchTracks;

  if (isMatchedToITS) {
    std::tie(itsFile, itsTree) = openTreeOrNull(fmt::format("{}/{}", dir, itsTracksFileName), "o2sim");
    std::tie(matchFile, matchTree) = openTreeOrNull(fmt::format("{}/{}", dir, matchFileName), "matchTPCITS");
    if (!itsTree || !matchTree) {
      return;
    }
    itsTree->SetBranchAddress("ITSTrack", &itsTracksPtr);
    matchTree->SetBranchAddress("TPCITS", &matchTracksPtr);
  }

  std::unique_ptr<TFile> tfIDFile;
  std::unique_ptr<TTree> tfIDTree;
  o2::dataformats::TFIDInfo* tfIDInfo{nullptr};
  Long64_t timeStamp = runNumberOrTimeStamp;
  const auto tfIDFullName = fmt::format("{}/{}", dir, tfIDFileName);
  if (fs::exists(tfIDFullName)) {
    std::tie(tfIDFile, tfIDTree) = openTreeOrNull(tfIDFullName, "tfidTree");
    if (tfIDTree) {
      tfIDTree->SetBranchAddress("tfidinfo", &tfIDInfo);
      tfIDTree->SetBranchAddress("ts", &timeStamp);
      LOGP(info, "Using per-time-frame CCDB time stamps from {}", tfIDFullName);
    }
  }

  const auto clName = fmt::format("{}/{}", dir, clusterNativeFileName);
  if (!fs::exists(clName)) {
    LOGP(error, "Cluster file {} does not exist", clName);
    return;
  }
  ClusterNativeHelper::Reader tpcClusterReader{};
  tpcClusterReader.init(clName.data());
  if (tpcClusterReader.getTreeSize() == 0) {
    LOGP(error, "Could not read a native cluster tree from {}", clName);
    return;
  }

  o2::utils::TreeStreamRedirector stream(outFile.data(), "recreate");

  ClusterNativeAccess clusterIndex{};
  std::unique_ptr<ClusterNative[]> clusterBuffer{};
  ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));

  long long nEvents = tpcTree->GetEntriesFast();
  if (isMatchedToITS) {
    nEvents = std::min<long long>(nEvents, std::min(itsTree->GetEntriesFast(), matchTree->GetEntriesFast()));
  }
  if (tfIDTree && tfIDTree->GetEntriesFast() < nEvents) {
    LOGP(error, "tfIDInfo tree has fewer entries ({}) than the data trees ({}); ignoring it and using runNumberOrTimeStamp for all events",
         tfIDTree->GetEntriesFast(), nEvents);
    tfIDTree.reset();
    tfIDFile.reset();
  }
  if (maxEvents >= 0 && maxEvents < nEvents) {
    LOGP(info, "Limiting to the first {} of {} available events (maxEvents)", maxEvents, nEvents);
    nEvents = maxEvents;
  }

  for (long long iEvent = 0; iEvent < nEvents; iEvent++) {
    tpcTree->GetEntry(iEvent);
    if (isMC && tpcMCTruth.size() != tpcTracks.size()) {
      LOGP(error, "TPCTracksMCTruth size ({}) does not match TPCTracks size ({}) for event {}, skipping event",
           tpcMCTruth.size(), tpcTracks.size(), iEvent);
      continue;
    }
    tpcClusterReader.read(iEvent);
    tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);
    if (isMatchedToITS) {
      itsTree->GetEntry(iEvent);
      matchTree->GetEntry(iEvent);
    }
    if (tfIDTree) {
      tfIDTree->GetEntry(iEvent);
    }

    // setMembers()/loadCalibs/setRefit()... depend on tracks, clusters and timestamp, so they must be redone per event
    calcdEdx.setMembers(tpcTrackClIdxVecInput, clusterIndex, &tpcTracks);
    if (localCCDBFolder.empty()) {
      bool loadSCCorrMap = false;
      for (const auto& s : settingsList) {
        loadSCCorrMap |= (s.correctionMask & CorrectionFlags::dEdxSC) == CorrectionFlags::dEdxSC;
      }
      calcdEdx.loadCalibsFromCCDB(timeStamp, isMC, loadSCCorrMap, applySCToRefitTransform && useRefit, applyVDriftToRefitTransform && useRefit);
    } else {
      calcdEdx.loadCalibsFromLocalCCDBFolder(localCCDBFolder.data());
    }
    if (useRefit) {
      calcdEdx.setRefit();
    }

    const size_t nSelectable = isMatchedToITS ? matchTracks.size() : tpcTracks.size();
    LOGP(info, "Processing event {} with {} {} and {} settings", iEvent, nSelectable, isMatchedToITS ? "matched tracks" : "tracks", settingsList.size());

    std::vector<TrackTPC> tpcOut;
    std::vector<o2::its::TrackITS> itsTracksOut;
    std::vector<o2::dataformats::TrackTPCITS> matchTracksOut;
    std::vector<std::vector<dEdxInfo>> dEdxOut; // [track][settings]
    std::vector<AverageOccupancy> averageOccOut;
    std::vector<o2::MCCompLabel> mcLabelOut;

    for (size_t i = 0; i < nSelectable; i++) {
      size_t tpcIndex = i;
      if (isMatchedToITS) {
        const auto& itstpc = matchTracks[i];
        if (itstpc.getRefITS().getSource() != o2::dataformats::GlobalTrackID::ITS) {
          continue;
        }
        tpcIndex = itstpc.getRefTPC().getIndex();
        itsTracksOut.emplace_back(itsTracks[itstpc.getRefITS().getIndex()]);
        matchTracksOut.emplace_back(itstpc);
      }

      TrackTPC track(tpcTracks[tpcIndex]); // local copy: refit/propagation inside calculatedEdxMultipleSettings mutate the track in place
      std::vector<dEdxInfo> dEdxVec;
      AverageOccupancy averageOcc;
      calcdEdx.calculatedEdxMultipleSettings(track, dEdxVec, averageOcc, activeSettingsList, isMC ? &tpcMCTruth[tpcIndex] : nullptr);

      tpcOut.emplace_back(track);
      dEdxOut.emplace_back(std::move(dEdxVec));
      averageOccOut.emplace_back(averageOcc);
      if (isMC) {
        mcLabelOut.emplace_back(tpcMCTruth[tpcIndex]);
      }
    }

    // per-event summary: refit/propagation failures and how many row gaps were filled as subthreshold clusters per settingsList entry
    const long nPropagationFailed = calcdEdx.getNPropagationFailed();
    const long nRowsProcessed = calcdEdx.getNRowsProcessed();
    const auto& nSubThresholdFilledPerSettings = calcdEdx.getNSubThresholdFilledPerSettings();
    calcdEdx.resetDebugCounters();
    std::string subThresholdBreakdown;
    for (size_t i = 0; i < nSubThresholdFilledPerSettings.size(); i++) {
      subThresholdBreakdown += fmt::format("{}dEdx{}={}", i > 0 ? ", " : "", i, nSubThresholdFilledPerSettings[i]);
    }
    LOGP(info, "Event {}: refit/propagation failed for {}/{} rows ({:.2f}%); gap-cluster(s) filled as subthreshold per settings entry: {}",
         iEvent, nPropagationFailed, nRowsProcessed, nRowsProcessed > 0 ? 100. * nPropagationFailed / nRowsProcessed : 0., subThresholdBreakdown);

    // one row per track, with one "dEdx<iSettings>" branch per entry in settingsList
    for (size_t i = 0; i < dEdxOut.size(); i++) {
      auto& row = stream << "tree"
                         << "iEvent=" << iEvent
                         << "timeStamp=" << timeStamp
                         << "tpc=" << tpcOut[i]
                         << "averageOcc=" << averageOccOut[i];
      for (size_t iSettings = 0; iSettings < settingsList.size(); iSettings++) {
        row << fmt::format("dEdx{}=", iSettings).c_str() << dEdxOut[i][iSettings];
      }
      if (tfIDTree) {
        row << "tfIDInfo=" << tfIDInfo;
      }
      if (isMatchedToITS) {
        row << "its=" << itsTracksOut[i]
            << "itstpc=" << matchTracksOut[i];
      }
      if (isMC) {
        const auto& label = mcLabelOut[i];
        row << "mcLabel=" << label;
      }
      row << "\n";
    }
  }

  stream.Close();

  const std::clock_t c_end = std::clock();
  const auto t_end = std::chrono::high_resolution_clock::now();

  std::cout << std::fixed << std::setprecision(2)
            << "CPU time used: "
            << (1000.0 * (c_end - c_start) / CLOCKS_PER_SEC) / 60000.0 << " minutes\n"
            << "Wall clock time passed: "
            << std::chrono::duration<double, std::milli>(t_end - t_start).count() / 60000.0 << " minutes\n";
}
