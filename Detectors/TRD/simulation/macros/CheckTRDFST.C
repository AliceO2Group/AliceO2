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

/// \file CheckTRDFST.C
/// \brief Simple macro to check TRD digits and tracklets post sim to post reconstruction

// a couple of steps are not included in this:
// It is assumed that the raw-to-tf is run in the directory you run the fst.
// you reconstruct to trddigits and trdtracklets in the fst/raw/timeframe directory
//
// alienv enter O2PDPSuite/latest-o2 Readout/latest-o2
// DISABLE_PROCESSING=1 NEvents=20 NEventsQED=100 SHMSIZE=128000000000 TPCTRACKERSCRATCHMEMORY=40000000000 SPLITTRDDIGI=0 GENERATE_ITSMFT_DICTIONARIES=1 $O2_ROOT/prodtests/full_system_test.sh
// $O2_ROOT/prodtests/full-system-test/convert-raw-to-tf-file.sh
// cd raw/timeframe
// o2-raw-tf-reader-workflow --input-data o2_rawtf_run00000000_tf00000001_???????.tf | o2-trd-datareader --fixsm1617 --enable-root-output | o2-dpl-run --run -b
// Then run this script.
// the convert-raw-to-tf-file.sh must be run on a machine with >200G the rest can be run anywhere.
//
// WARNING: This should be used only for rather small data sets (e.g. the 20 PbPb events as suggested above).
//          Otherwise, due to the quadratic dependency on the number of digits/tracklets this takes very long,
//          especially in case there are no matches found.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TLegend.h>

#include "FairLogger.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#endif

using namespace o2::trd;

void CheckTRDFST(bool ignoreTrkltPid = true, std::string fstbasedir = "./",
                 std::string digitfile = "trddigits.root", std::string trackletfile = "trdtracklets.root",
                 std::string recodigitfile = "trddigits.root", std::string recotrackletfile = "trdtracklets.root")
{
  TFile* dfin = TFile::Open(Form("%s%s", fstbasedir.data(), digitfile.data()));
  TTree* digitTree = (TTree*)dfin->Get("o2sim");
  std::vector<Digit>* digits = nullptr;
  digitTree->SetBranchAddress("TRDDigit", &digits);
  int ndigitev = digitTree->GetEntries();

  TFile* tfin = TFile::Open(Form("%s%s", fstbasedir.data(), trackletfile.data()));
  TTree* trackletTree = (TTree*)tfin->Get("o2sim");
  std::vector<Tracklet64>* tracklets = nullptr;
  trackletTree->SetBranchAddress("Tracklet", &tracklets);
  int ntrackletev = trackletTree->GetEntries();

  TFile* dfinreco = TFile::Open(Form("%s/raw/timeframe/%s", fstbasedir.data(), recodigitfile.data()));
  TTree* digitTreereco = (TTree*)dfinreco->Get("o2sim");
  std::vector<Digit>* digitsreco = nullptr;
  digitTreereco->SetBranchAddress("TRDDigit", &digitsreco);
  int ndigitevreco = digitTreereco->GetEntries();

  TFile* tfinreco = TFile::Open(Form("%s/raw/timeframe/%s", fstbasedir.data(), recotrackletfile.data()));
  TTree* trackletTreereco = (TTree*)tfinreco->Get("o2sim");
  std::vector<Tracklet64>* trackletsreco = nullptr;
  trackletTreereco->SetBranchAddress("Tracklet", &trackletsreco);
  int ntrackletevreco = trackletTreereco->GetEntries();

  if ((ndigitev != ntrackletev) || (ndigitevreco != ntrackletevreco) || (ndigitev != ndigitevreco)) {
    // make sure the number of entries is the same in all trees
    LOG(error) << "The trees have a different number of entries";
  }

  for (int iev = 0; iev < ntrackletev; ++iev) {
    digitTree->GetEvent(iev);
    trackletTree->GetEvent(iev);
    digitTreereco->GetEvent(iev);
    trackletTreereco->GetEvent(iev);

    // compare tracklets
    if (tracklets->size() != trackletsreco->size()) {
      LOG(warn) << "Number of tracklets does not match for entry " << iev;
    }
    std::vector<bool> flags(trackletsreco->size());
    for (size_t iTrklt = 0; iTrklt < tracklets->size(); ++iTrklt) {
      const auto& trklt = tracklets->at(iTrklt);
      auto tw = trklt.getTrackletWord();
      tw |= (0xfUL << 60); // set all format bits
      if (ignoreTrkltPid) {
        tw |= 0xffffffUL; // set all PID bits
      }
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
          break;
        }
      }
    }

    // compare digits
    if (digits->size() != digitsreco->size()) {
      LOG(warn) << "Number of digits does not match for entry " << iev;
    }
    std::vector<bool> flagsDigit(digitsreco->size());
    for (size_t iDigit = 0; iDigit < digits->size(); ++iDigit) {
      const auto& digit = digits->at(iDigit);
      for (size_t iDigitReco = 0; iDigitReco < digitsreco->size(); ++iDigitReco) {
        if (flagsDigit[iDigitReco]) {
          continue;
        }
        const auto& digitReco = digitsreco->at(iDigitReco);
        if (digitReco == digit) {
          flagsDigit[iDigitReco] = true;
          break;
        }
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
