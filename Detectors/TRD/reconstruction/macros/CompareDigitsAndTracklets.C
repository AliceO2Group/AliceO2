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

/// \file CompareDigitsAndTracklets.C
/// \brief Simple macro to compare TRD digits and tracklets e.g. before and after conversion to RAW

// see the Detectors/TRD/reconstruction/README.md for the steps required to convert a simulation to RAW and read it back
//
// WARNING: This should be used only for rather small data sets (e.g. the 20 PbPb events as suggested above).
//          Otherwise, due to the quadratic dependency on the number of digits/tracklets this takes very long.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>

#include "FairLogger.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#endif

using namespace o2::trd;

void CompareDigitsAndTracklets(bool ignoreTrkltPid = false,
                               std::string digitfile = "trddigitsOrig.root",
                               std::string trackletfile = "trdtrackletsOrig.root",
                               std::string recodigitfile = "trddigits.root",
                               std::string recotrackletfile = "trdtracklets.root")
{
  TFile* dfin = TFile::Open(digitfile.c_str());
  TTree* digitTree = (TTree*)dfin->Get("o2sim");
  std::vector<Digit>* digits = nullptr;
  digitTree->SetBranchAddress("TRDDigit", &digits);
  int ndigitev = digitTree->GetEntries();

  TFile* tfin = TFile::Open(trackletfile.c_str());
  TTree* trackletTree = (TTree*)tfin->Get("o2sim");
  std::vector<Tracklet64>* tracklets = nullptr;
  trackletTree->SetBranchAddress("Tracklet", &tracklets);
  std::vector<TriggerRecord>* trigRecs = nullptr;
  trackletTree->SetBranchAddress("TrackTrg", &trigRecs);
  int ntrackletev = trackletTree->GetEntries();

  TFile* dfinreco = TFile::Open(recodigitfile.c_str());
  TTree* digitTreereco = (TTree*)dfinreco->Get("o2sim");
  std::vector<Digit>* digitsreco = nullptr;
  digitTreereco->SetBranchAddress("TRDDigit", &digitsreco);
  int ndigitevreco = digitTreereco->GetEntries();

  TFile* tfinreco = TFile::Open(recotrackletfile.c_str());
  TTree* trackletTreereco = (TTree*)tfinreco->Get("o2sim");
  std::vector<Tracklet64>* trackletsreco = nullptr;
  trackletTreereco->SetBranchAddress("Tracklet", &trackletsreco);
  std::vector<TriggerRecord>* trigRecsReco = nullptr;
  trackletTreereco->SetBranchAddress("TrackTrg", &trigRecsReco);
  int ntrackletevreco = trackletTreereco->GetEntries();

  if ((ndigitev != ntrackletev) || (ndigitevreco != ntrackletevreco) || (ndigitev != ndigitevreco)) {
    // make sure the number of entries is the same in all trees
    LOG(error) << "The trees have a different number of entries";
    return;
  }

  for (int iev = 0; iev < ntrackletev; ++iev) {
    digitTree->GetEvent(iev);
    trackletTree->GetEvent(iev);
    digitTreereco->GetEvent(iev);
    trackletTreereco->GetEvent(iev);

    // compare trigger records
    if (trigRecs->size() != trigRecsReco->size()) {
      LOG(warn) << "Number of trigger records does not match for entry " << iev;
      continue;
    }
    for (size_t iTrig = 0; iTrig < trigRecs->size(); ++iTrig) {
      const auto& trig = trigRecs->at(iTrig);
      const auto& trigReco = trigRecsReco->at(iTrig);
      if (!(trig == trigReco)) {
        LOGF(error, "Trigger records don't match at trigger %lu. Reference orbit/bc (%u/%u), orbit/bc (%u/%u)",
             iTrig, trig.getBCData().orbit, trig.getBCData().bc, trigReco.getBCData().orbit, trigReco.getBCData().bc);
        break;
      }
    }

    // compare tracklets
    if (tracklets->size() != trackletsreco->size()) {
      LOG(warn) << "Number of tracklets does not match for entry " << iev;
      continue;
    }
    std::vector<bool> flags(trackletsreco->size());
    for (size_t iTrklt = 0; iTrklt < tracklets->size(); ++iTrklt) {
      const auto& trklt = tracklets->at(iTrklt);
      auto tw = trklt.getTrackletWord();
      tw |= (0xfUL << 60); // set all format bits
      if (ignoreTrkltPid) {
        tw |= 0xffffffUL; // set all PID bits
      }
      bool hasMatch = false;
      for (size_t iTrkltReco = 0; iTrkltReco < trackletsreco->size(); ++iTrkltReco) {
        if (flags[iTrkltReco]) {
          // tracklet has already been matched, skip it
          continue;
        }
        const auto& trkltReco = trackletsreco->at(iTrkltReco);
        auto twReco = trkltReco.getTrackletWord();
        twReco |= (0xfUL << 60); // set all format bits
        if (ignoreTrkltPid) {
          twReco |= 0xffffffUL; // set all PID bits
        }
        if (tw == twReco) {
          // matching tracklet found
          flags[iTrkltReco] = true;
          hasMatch = true;
          break;
        }
      }
      if (!hasMatch) {
        LOGF(error, "No match for reference tracklet at index %lu\n", iTrklt);
        break;
      }
    }

    // compare digits
    if (digits->size() != digitsreco->size()) {
      LOG(warn) << "Number of digits does not match for entry " << iev;
      continue;
    }
    std::vector<bool> flagsDigit(digitsreco->size());
    for (size_t iDigit = 0; iDigit < digits->size(); ++iDigit) {
      bool hasMatch = false;
      const auto& digit = digits->at(iDigit);
      for (size_t iDigitReco = 0; iDigitReco < digitsreco->size(); ++iDigitReco) {
        if (flagsDigit[iDigitReco]) {
          continue;
        }
        const auto& digitReco = digitsreco->at(iDigitReco);
        if (digitReco == digit) {
          flagsDigit[iDigitReco] = true;
          hasMatch = true;
          break;
        }
      }
      if (!hasMatch) {
        LOGF(error, "No match for reference digit at index %lu\n", iDigit);
        break;
      }
    }

    // summary for this TF
    int matchedTracklets = 0;
    for (auto f : flags) {
      if (f) {
        matchedTracklets++;
      }
    }
    int matchedDigits = 0;
    for (auto f : flagsDigit) {
      if (f) {
        matchedDigits++;
      }
    }
    LOGF(info, "Number of tracklets in simulation: %lu. In reco: %lu. Number of tracklets from simulation for which a match in reco was found: %i", tracklets->size(), trackletsreco->size(), matchedTracklets);
    LOGF(info, "Number of digits in simulation: %lu. In reco: %lu. Number of digits from simulation for which a match in reco was found: %i", digits->size(), digitsreco->size(), matchedDigits);
  }

  if (ignoreTrkltPid) {
    LOG(warn) << "The PID values stored inside the tracklets have been ignored for this comparison";
  }
}
