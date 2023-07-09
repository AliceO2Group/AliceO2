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

/*
  Provided an input file with tracklets and trigger records this macro produces a bitset
  where noisy MCMs are flagged.
  In case useMean is requested the mean number of tracklets sent from each MCM is calculated.
  MCMs which don't send any tracklets are not taken into account. The 10% of MCMs which send
  the highest number of tracklets are also not considered for the mean calculation. Then, MCMs
  which sent more than 100 times as many tracklets as the mean are flagged as noisy. If useMean
  is not used simply the MCMs are flagged which sent on average at least one tracklet every 10th
  trigger.

  Important: Compare the number of tracklets sent from the first and the last masked MCM. In case
  there is a large discrepancy between the two numbers the noise threshold might need to be adjusted
  or not enough statistics is used.
*/

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "DataFormatsTRD/NoiseCalibration.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/HelperMethods.h"

#include <TFile.h>
#include <TTree.h>

#include <string>
#include <vector>
#include <set>
#include <utility>
#include <map>
#include <algorithm>
#include <locale.h>

#endif

void checkNoisyMCMs(std::string inpFile = "trdtracklets.root", bool useMean = true, bool detailedOutput = true, bool enableWriteCCDB = false)
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

  std::array<std::pair<unsigned int, unsigned int>, constants::MAXHALFCHAMBER * constants::NMCMHCMAX> trackletCounter{};
  std::for_each(trackletCounter.begin(), trackletCounter.end(), [](auto& counter) { static int idx = 0; counter.second = idx++; });

  // count number of tracklets per MCM
  size_t totalTrackletCounter = 0;
  size_t totalTriggerCounter = 0;
  for (int iEntry = 0; iEntry < tree->GetEntries(); ++iEntry) {
    tree->GetEntry(iEntry);
    totalTriggerCounter += trigIn.size();
    for (const auto& tracklet : tracklets) {
      totalTrackletCounter++;
      int mcmGlb = NoiseStatusMCM::getMcmIdxGlb(tracklet.getHCID(), tracklet.getROB(), tracklet.getMCM());
      trackletCounter[mcmGlb].first++;
    }
  }

  // estimate noise threshold
  std::sort(trackletCounter.begin(), trackletCounter.end(), [](const auto& a, const auto& b) { return a.first < b.first; }); // sort by number of tracklets per MCM
  float mean90 = 0;                                                                                                          // mean excluding 10% of MCMs with most entries
  int nActiveMcms90 = 0;
  std::vector<int> nTrkltsPerActiveMcm; // for median calculation
  for (int idx = 0; idx < constants::MAXHALFCHAMBER * constants::NMCMHCMAX; ++idx) {
    if (trackletCounter[idx].first > 0) {
      nTrkltsPerActiveMcm.push_back(trackletCounter[idx].first);
      if (idx < static_cast<int>(0.9 * (constants::MAXHALFCHAMBER * constants::NMCMHCMAX))) {
        // for the average number of tracklets we exclude the 10% of MCMs with most entries
        nActiveMcms90++;
        mean90 += trackletCounter[idx].first;
      }
    }
  }
  if (!nActiveMcms90) {
    printf("ERROR: did not find any MCMs which sent tracklets. Aborting\n");
    return;
  }

  auto n = nTrkltsPerActiveMcm.size() / 2;
  auto median = nTrkltsPerActiveMcm[n]; // the distinction between odd/even number of entries in the vector is irrelevant
  mean90 /= nActiveMcms90;
  auto noiseThreshold = (useMean) ? 30.f * mean90 : (float)totalTriggerCounter / 5.f;

  setlocale(LC_NUMERIC, "en_US.utf-8");
  printf("Info: Found in total %'lu MCMs which sent tracklets with a median of %i tracklets per MCM\n", nTrkltsPerActiveMcm.size(), median);
  printf("Info: Excluding the 10%% of MCMs which sent the highest number of tracklets %'i MCMs remain with on average %.2f tracklets per MCM\n", nActiveMcms90, mean90);
  printf("Info: Checked %lu triggers in total\n", totalTriggerCounter);
  printf("Important: Masking MCMs which sent more than %.2f tracklets for given period\n", noiseThreshold);

  std::vector<int> nTrackletsFromNoisyMcm;
  bool hasPrintedFirstNoisy = false;
  for (int idx = 0; idx < constants::MAXHALFCHAMBER * constants::NMCMHCMAX; ++idx) {
    auto mcmIdx = trackletCounter[idx].second;
    if (trackletCounter[idx].first > noiseThreshold) {
      if (!hasPrintedFirstNoisy) {
        printf("Info: The first masked MCM idx(%i) with glb idx %i sent %i trackelts\n", idx, mcmIdx, trackletCounter[idx].first);
        hasPrintedFirstNoisy = true;
      }
      noiseMap.setIsNoisy(mcmIdx);
      nTrackletsFromNoisyMcm.push_back(trackletCounter[idx].first);
    }
  }
  if (noiseMap.getNumberOfNoisyMCMs() > 0) {
    printf("Info: Last masked MCM sent %i tracklets\n", nTrackletsFromNoisyMcm.back());
  }

  if (enableWriteCCDB) {
    printf("Info: Uploading to CCDB (by default only to ccdb-test)\n");
    o2::ccdb::CcdbApi ccdb;
    ccdb.init("http://ccdb-test.cern.ch:8080");
    std::map<std::string, std::string> metadata;
    metadata.emplace(std::make_pair("UploadedBy", "marten"));
    auto timeStampStart = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto timeStampEnd = timeStampStart;
    timeStampEnd += 1e3 * 60 * 60 * 24 * 60; // 60 days
    ccdb.storeAsTFileAny(&noiseMap, "TRD/Calib/NoiseMapMCM", metadata, timeStampStart, timeStampEnd);
  } else {
    // write to local file
    printf("Info: Writing to local file mcmNoiseMap.root\n");
    auto fOut = new TFile("mcmNoiseMap.root", "recreate");
    fOut->WriteObjectAny(&noiseMap, "o2::trd::NoiseStatusMCM", "map");
    fOut->Close();
  }

  printf("Info: Found in total %lu noisy MCMs for %'lu tracklets from %'lu triggers\n", noiseMap.getNumberOfNoisyMCMs(), totalTrackletCounter, totalTriggerCounter);

  std::set<int> channelsToBeMasked;
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  auto channelInfos = ccdbmgr.get<o2::trd::ChannelInfoContainer>("TRD/Calib/ChannelStatus");
  if (detailedOutput) {
    int countFoundMatch = 0;
    int countFoundNoMatch = 0;
    size_t countTrackletsFromNoisyMcms = 0;
    printf("Info: Number of tracklets sent per masked MCM: \n");
    printf("----->\n");
    for (int idx = 0; idx < constants::MAXHALFCHAMBER * constants::NMCMHCMAX; ++idx) {
      auto mcmIdx = trackletCounter[idx].second;
      if (noiseMap.getIsNoisy(mcmIdx)) {
        bool partnerFound = false;
        countTrackletsFromNoisyMcms += trackletCounter[idx].first;
        int hcid, rob, mcm;
        NoiseStatusMCM::convertMcmIdxGlb(mcmIdx, hcid, rob, mcm);
        printf("Masked MCM idx (%i), glbIdx(%i), HCID(%i), ROB(%i), MCM(%i). nTracklets: %i\n", idx, mcmIdx, hcid, rob, mcm, trackletCounter[idx].first);
        for (int iCh = 0; iCh < constants::NADCMCM; ++iCh) {
          auto chIdx = HelperMethods::getGlobalChannelIndex(hcid / 2, rob, mcm, iCh);
          auto ch = channelInfos->getChannel(chIdx);
          if (ch.getMean() > 20 || ch.getRMS() > 15) {
            channelsToBeMasked.insert(chIdx);
            ++countFoundMatch;
            partnerFound = true;
            printf("Found corresponding noisy channel %i: mean(%f), rms(%f)\n", iCh, ch.getMean(), ch.getRMS());
          }
        }
        if (!partnerFound) {
          ++countFoundNoMatch;
          for (int iCh = 0; iCh < constants::NADCMCM; ++iCh) {
            auto chIdx = HelperMethods::getGlobalChannelIndex(hcid / 2, rob, mcm, iCh);
            auto ch = channelInfos->getChannel(chIdx);
            // printf("No matching noisy channel found. Printing channel %i: mean(%f), rms(%f)\n", iCh, ch.getMean(), ch.getRMS());
          }
        }
      }
    }
    printf("\n<-----\n");
    printf("Info: Number of tracklets sent from masked MCMs: %'lu (%.2f%%)\n", countTrackletsFromNoisyMcms, (float)countTrackletsFromNoisyMcms / totalTrackletCounter * 100);
    printf("Found %i with matching noisy channel and %i without\n", countFoundMatch, countFoundNoMatch);
  }
  printf("Info: Done\n");
  printf("Listing channels to be masked:\n");
  for (auto chIdx : channelsToBeMasked) {
    int det, rob, mcm, channel;
    HelperMethods::getPositionFromGlobalChannelIndex(chIdx, det, rob, mcm, channel);
    int sec = HelperMethods::getSector(det);
    int stack = HelperMethods::getStack(det);
    int layer = HelperMethods::getLayer(det);
    auto ch = channelInfos->getChannel(chIdx);
    LOGP(info, "{}_{}_{}: ROB({}), MCM({}), channel({}); Measured ADC values from noise run: mean {}, rms {}, nEntries {}",
         sec, stack, layer, rob, mcm, channel, ch.getMean(), ch.getRMS(), ch.getEntries());
  }
}
