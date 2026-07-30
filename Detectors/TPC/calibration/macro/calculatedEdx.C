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
#include <thread>
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
#include "SimulationDataFormat/MCTrack.h"
#include "Steer/MCKinematicsReader.h"
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
} // namespace

/// \param dir directory containing tpctracks.root, tpc-native-clusters.root and, if isMatchedToITS, o2trac_its.root/o2match_itstpc.root
/// \param runNumberOrTimeStamp run number or timestamp used to load the calibration objects from CCDB for every timeframe, overridden per timeframe whenever a tfIDFileName file is found in dir
/// \param outFile name of the output file with the calculated dE/dx tree
/// \param localCCDBFolder if non-empty, load calibration objects from local CCDB snapshots in this folder instead of from the CCDB server
/// \param correctionMask which corrections to apply
/// \param clusterMask which clusters to exclude from the dE/dx calculation
/// \param lowTruncation lower cluster cut of the truncated mean, e.g. 0.015 keeps clusters starting from the 1.5% percentile of sorted charges
/// \param highTruncation higher cluster cut of the truncated mean, e.g. 0.6 keeps clusters up to the 60% percentile of sorted charges
/// \param subthresholdMethod how to fill subthreshold clusters: 0 = minimum charge, 1 = minimum charge / 2
/// \param stackBoundaryMethod exclude clusters on GEM stack boundary rows from the dE/dx calculation: 0 = disabled, 1 = exclude the boundary row, 2 = also exclude the row next to the boundary
/// \param useRefit refit the tracks at each cluster row using the GPU refitter (default)
/// \param propagateTrack propagate the full track, including material corrections, instead of refitting; only used if useRefit is false
/// \param propagateParams propagate only the track parameters (fastest option, no material corrections); only used if useRefit and propagateTrack are both false
/// \param debug enable the CalculatedEdx debug streamer, additionally writing one dEdxDebug_t<i>.root file with per-cluster information per worker thread
/// \param nThreads number of worker threads used to process the tracks of one event in parallel
/// \param isMC set to true for MC productions to read the true MC track (TPCTracksMCTruth branch + mcKineFileName) for each track into an additional "mctrack" branch
/// \param isMatchedToITS set to true to also read o2trac_its.root/o2match_itstpc.root and restrict the dE/dx calculation to TPC tracks matched to an ITS track
/// \param tfIDFileName name of the optional timeframe ID file; if found in dir, its per-entry time stamp is used for the CCDB instead of runNumberOrTimeStamp
/// \param tpcTracksFileName name of the file with the TPC tracks
/// \param clusterNativeFileName name of the file with the TPC native clusters
/// \param itsTracksFileName name of the file with the ITS tracks; only used if isMatchedToITS is true
/// \param matchFileName name of the file with the TPC-ITS match information; only used if isMatchedToITS is true
/// \param mcKineFileName base name (without extension) of the MC kinematics file used to resolve MC track labels; only used if isMC is true
void calculatedEdx(const std::string dir = ".",
                   const long runNumberOrTimeStamp = 0,
                   const std::string outFile = "dEdxCalc.root",
                   const std::string localCCDBFolder = "",
                   const CorrectionFlags correctionMask = CorrectionFlags::TopologyPol | CorrectionFlags::dEdxResidual,
                   const ClusterFlags clusterMask = ClusterFlags::ExcludeEdgeCl,
                   const float lowTruncation = 0.015f,
                   const float highTruncation = 0.6f,
                   const int subthresholdMethod = 0,
                   const int stackBoundaryMethod = 0,
                   const bool useRefit = true,
                   const bool propagateTrack = false,
                   const bool propagateParams = false,
                   const bool debug = false,
                   const bool isMC = false,
                   const bool isMatchedToITS = false,
                   const size_t nThreads = 8,
                   const std::string tfIDFileName = "o2_tfidinfo.root",
                   const std::string tpcTracksFileName = "tpctracks.root",
                   const std::string clusterNativeFileName = "tpc-native-clusters.root",
                   const std::string itsTracksFileName = "o2trac_its.root",
                   const std::string matchFileName = "o2match_itstpc.root",
                   const std::string mcKineFileName = "o2sim")
{
  ROOT::EnableThreadSafety();

  if (nThreads == 0) {
    LOGP(error, "nThreads must be at least 1");
    return;
  }
  if (lowTruncation < 0.f || highTruncation > 1.f || lowTruncation >= highTruncation) {
    LOGP(error, "Invalid truncation range [{}, {}); expected 0 <= lowTruncation < highTruncation <= 1", lowTruncation, highTruncation);
    return;
  }
  if (subthresholdMethod != 0 && subthresholdMethod != 1) {
    LOGP(error, "Invalid subthresholdMethod {}; expected 0 (minimum charge) or 1 (minimum charge / 2)", subthresholdMethod);
    return;
  }
  if (stackBoundaryMethod < 0 || stackBoundaryMethod > 2) {
    LOGP(error, "Invalid stackBoundaryMethod {}; expected 0 (disabled), 1 (exclude boundary row) or 2 (also exclude the adjacent row)", stackBoundaryMethod);
    return;
  }

  const std::clock_t c_start = std::clock();
  const auto t_start = std::chrono::high_resolution_clock::now();

  // one CalculatedEdx per worker thread
  std::vector<CalculatedEdx> calcdEdxPerThread(nThreads);
  for (auto& calcdEdx : calcdEdxPerThread) {
    calcdEdx.setDebug(debug);
    calcdEdx.setPropagateTrack(propagateTrack);
    calcdEdx.setPropagateParams(propagateParams);
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
  o2::steer::MCKinematicsReader mcReader;
  if (isMC) {
    if (!tpcTree->GetBranch("TPCTracksMCTruth")) {
      LOGP(error, "Branch 'TPCTracksMCTruth' not found in {}/{}, cannot resolve MC truth", dir, tpcTracksFileName);
      return;
    }
    tpcTree->SetBranchAddress("TPCTracksMCTruth", &tpcMCTruthPtr);
    const auto kineFileName = fmt::format("{}/{}", dir, mcKineFileName);
    if (!mcReader.initFromKinematics(kineFileName)) {
      LOGP(error, "Could not initialize MCKinematicsReader from {}", kineFileName);
      return;
    }
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

    // setMembers()/loadCalibs/setRefit()... depend on tracks, clusters and timestamp, so they must be redone per event, per thread instance
    for (auto& calcdEdx : calcdEdxPerThread) {
      calcdEdx.setMembers(tpcTrackClIdxVecInput, clusterIndex, &tpcTracks);
      if (localCCDBFolder.empty()) {
        calcdEdx.loadCalibsFromCCDB(timeStamp, isMC);
      } else {
        calcdEdx.loadCalibsFromLocalCCDBFolder(localCCDBFolder.data());
      }
      if (useRefit) {
        calcdEdx.setRefit();
      }
    }

    const size_t nSelectable = isMatchedToITS ? matchTracks.size() : tpcTracks.size();
    LOGP(info, "Processing event {} with {} {} using {} threads", iEvent, nSelectable, isMatchedToITS ? "matched tracks" : "tracks", nThreads);

    std::vector<std::vector<TrackTPC>> tpcOut(nThreads);
    std::vector<std::vector<o2::its::TrackITS>> itsTracksOut(nThreads);
    std::vector<std::vector<o2::dataformats::TrackTPCITS>> matchTracksOut(nThreads);
    std::vector<std::vector<dEdxInfo>> dEdxOut(nThreads);
    std::vector<std::vector<AverageOccupancy>> averageOccOut(nThreads);
    std::vector<std::vector<o2::MCCompLabel>> mcLabelOut(nThreads);

    const size_t chunkSize = (nSelectable + nThreads - 1) / nThreads;
    std::vector<std::thread> threads;
    for (size_t iThread = 0; iThread < nThreads; iThread++) {
      const size_t start = iThread * chunkSize;
      const size_t end = std::min(start + chunkSize, nSelectable);
      if (start >= end) {
        continue;
      }
      threads.emplace_back([&, iThread, start, end]() {
        auto& calcdEdx = calcdEdxPerThread[iThread];
        const std::string debugFile = fmt::format("dEdxDebug_t{}.root", iThread);
        for (size_t i = start; i < end; i++) {
          size_t tpcIndex = i;
          if (isMatchedToITS) {
            const auto& itstpc = matchTracks[i];
            if (itstpc.getRefITS().getSource() != o2::dataformats::GlobalTrackID::ITS) {
              continue;
            }
            tpcIndex = itstpc.getRefTPC().getIndex();
            itsTracksOut[iThread].emplace_back(itsTracks[itstpc.getRefITS().getIndex()]);
            matchTracksOut[iThread].emplace_back(itstpc);
          }

          TrackTPC track(tpcTracks[tpcIndex]); // local copy: setRefit()/propagation mutate the track in place
          dEdxInfo dEdx;
          AverageOccupancy averageOcc;
          calcdEdx.calculatedEdx(track, dEdx, averageOcc, lowTruncation, highTruncation, correctionMask, clusterMask, subthresholdMethod, stackBoundaryMethod, debugFile.c_str());

          tpcOut[iThread].emplace_back(track);
          dEdxOut[iThread].emplace_back(dEdx);
          averageOccOut[iThread].emplace_back(averageOcc);
          if (isMC) {
            mcLabelOut[iThread].emplace_back(tpcMCTruth[tpcIndex]);
          }
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }

    // write out sequentially in the main thread: no locking needed since all worker threads have already joined
    for (size_t iThread = 0; iThread < nThreads; iThread++) {
      for (size_t i = 0; i < dEdxOut[iThread].size(); i++) {
        auto& row = stream << "tree"
                           << "iEvent=" << iEvent
                           << "timeStamp=" << timeStamp
                           << "tpc=" << tpcOut[iThread][i]
                           << "dEdx=" << dEdxOut[iThread][i]
                           << "averageOcc=" << averageOccOut[iThread][i];
        if (tfIDTree) {
          row << "tfIDInfo=" << tfIDInfo;
        }
        if (isMatchedToITS) {
          row << "its=" << itsTracksOut[iThread][i]
              << "itstpc=" << matchTracksOut[iThread][i];
        }
        if (isMC) {
          const auto& label = mcLabelOut[iThread][i];
          const auto* mcTrack = label.isValid() ? mcReader.getTrack(label) : nullptr;
          row << "mcLabel=" << label;
          if (mcTrack) {
            row << "mctrack=" << *mcTrack;
          }
        }
        row << "\n";
      }
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
