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

/// \file checkNoisyMCMs.C
/// \brief macro identify noisy MCMs based on the number of found tracklets

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "DataFormatsTRD/NoiseCalibration.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"

#include <TFile.h>
#include <TTree.h>

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <algorithm>

#endif

void checkNoisyMCMs(std::string inpFile = "trdtracklets.root", bool enableWriteCCDB = false)
{
  using namespace o2::trd;

  NoiseStatusMCM noiseMap;

  auto fIn = TFile::Open(inpFile.c_str(), "open");
  if (!fIn) {
    printf("ERROR: could not open file %s\n", inpFile.c_str());
    return;
  }
  auto tree = (TTree*)fIn->Get("o2sim");
  if (!tree) {
    printf("ERROR: did not find tree with tracklets in file %s\n", inpFile.c_str());
    return;
  }

  std::vector<o2::trd::TriggerRecord> trigIn, *trigInPtr{&trigIn};
  std::vector<o2::trd::Tracklet64> tracklets, *trackletsInPtr{&tracklets};
  tree->SetBranchAddress("TrackTrg", &trigInPtr);
  tree->SetBranchAddress("Tracklet", &trackletsInPtr);

  std::array<std::pair<unsigned int, unsigned short>, constants::MAXHALFCHAMBER * constants::NMCMHCMAX> trackletCounter{};
  std::for_each(trackletCounter.begin(), trackletCounter.end(), [](auto& counter) { static int idx = 0; counter.second = idx++; });

  // count number of tracklets per MCM
  size_t totalTrackletCounter = 0;
  for (int iEntry = 0; iEntry < tree->GetEntries(); ++iEntry) {
    tree->GetEntry(iEntry);
    for (const auto& tracklet : tracklets) {
      totalTrackletCounter++;
      int mcmGlb = NoiseStatusMCM::getMcmIdxGlb(tracklet.getHCID(), tracklet.getROB(), tracklet.getMCM());
      trackletCounter[mcmGlb].first++;
    }
  }

  // estimate noise threshold
  std::sort(trackletCounter.begin(), trackletCounter.end(), [](const auto& a, const auto& b) { return a.first < b.first; }); // sort by number of tracklets per MCM
  float mean = 0;
  int nActiveMcms = 0;
  for (int idx = 0; idx < static_cast<int>(0.9 * (constants::MAXHALFCHAMBER * constants::NMCMHCMAX)); ++idx) {
    // get average number of tracklets discarding the 10% of MCMs with most entries
    if (trackletCounter[idx].first > 0) {
      nActiveMcms++;
      mean += trackletCounter[idx].first;
    }
  }
  if (!nActiveMcms) {
    printf("ERROR: did not find any MCMs which sent tracklets. Aborting\n");
    return;
  }
  mean /= nActiveMcms;
  float noiseThreshold = 10 * mean;

  for (const auto& counter : trackletCounter) {
    if (counter.first > noiseThreshold) {
      noiseMap.setIsNoisy(counter.second);
    }
  }

  if (enableWriteCCDB) {
    o2::ccdb::CcdbApi ccdb;
    ccdb.init("http://ccdb-test.cern.ch:8080");
    std::map<std::string, std::string> metadata;
    metadata.emplace(std::make_pair("UploadedBy", "marten"));
    auto timeStampStart = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto timeStampEnd = timeStampStart;
    timeStampEnd += 1e3 * 60 * 60 * 24 * 60; // 60 days
    ccdb.storeAsTFileAny(&noiseMap, "TRD/Calib/NoiseMapMCM", metadata, timeStampStart, timeStampEnd);
  }

  printf("Found in total %lu noisy MCMs for %lu tracklets from %lu trigger records\n", noiseMap.getNumberOfNoisyMCMs(), totalTrackletCounter, trigIn.size());
}
