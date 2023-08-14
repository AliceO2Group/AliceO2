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

/// \file PulseHeight.cxx
/// \brief Creates PH spectra from TRD digits found on tracks

#include "TRDCalibration/PulseHeight.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include <fairlogger/Logger.h>

using namespace o2::trd;
using namespace o2::trd::constants;

void PulseHeight::reset()
{
  mPHValues.clear();
  mDistances.clear();
}

void PulseHeight::init()
{
}

void PulseHeight::createOutputFile()
{
  mOutFile = std::make_unique<TFile>("trd_PH.root", "RECREATE");
  if (mOutFile->IsZombie()) {
    LOG(error) << "Failed to create output file!";
    return;
  }
  mOutTree = std::make_unique<TTree>("ph", "Data points for PH histograms");
  mOutTree->Branch("values", &mPHValuesPtr);
  mOutTree->Branch("dist", &mDistancesPtr);
  mWriteOutput = true;
  LOG(info) << "Writing PH data points to local file trd_PH.root";
}

void PulseHeight::closeOutputFile()
{
  if (!mWriteOutput) {
    return;
  }
  try {
    mOutFile->cd();
    mOutTree->Write();
    mOutTree.reset();
    mOutFile->Close();
    mOutFile.reset();
  } catch (std::exception const& e) {
    LOG(error) << "Failed to write output file, reason: " << e.what();
  }
  mWriteOutput = false;
}

void PulseHeight::setInput(const o2::globaltracking::RecoContainer& input, gsl::span<const Digit>* digits)
{
  mTracksInITSTPCTRD = input.getITSTPCTRDTracks<TrackTRD>();
  mTracksInTPCTRD = input.getTPCTRDTracks<TrackTRD>();
  mTrackletsRaw = input.getTRDTracklets();
  mTriggerRecords = input.getTRDTriggerRecords();
  mTrackTriggerRecordsTPCTRD = input.getTPCTRDTriggers();
  mTrackTriggerRecordsITSTPCTRD = input.getITSTPCTRDTriggers();
  mDigits = digits;
}

void PulseHeight::process()
{
  LOGP(debug, "Processing {} tracklets and {} digits from {} triggers and {} ITS-TPC-TRD tracks and {} TPC-TRD tracks",
       mTrackletsRaw.size(), mDigits->size(), mTriggerRecords.size(), mTracksInITSTPCTRD.size(), mTracksInTPCTRD.size());
  if (mDigits->size() == 0) {
    return;
  }
  size_t nTrkTrig[2] = {mTrackTriggerRecordsITSTPCTRD.size(), mTrackTriggerRecordsTPCTRD.size()};
  int lastTrkTrig[2] = {0};
  for (const auto& trig : mTriggerRecords) {
    if (trig.getNumberOfDigits() == 0) {
      continue;
    }
    for (int iTrackType = 0; iTrackType <= 1; ++iTrackType) {
      const gsl::span<const TrackTRD>* tracks = (iTrackType == 0) ? &mTracksInITSTPCTRD : &mTracksInTPCTRD;
      const gsl::span<const TrackTriggerRecord>* trackTriggers = (iTrackType == 0) ? &mTrackTriggerRecordsITSTPCTRD : &mTrackTriggerRecordsTPCTRD;
      for (int iTrigTrack = lastTrkTrig[iTrackType]; iTrigTrack < nTrkTrig[iTrackType]; ++iTrigTrack) {
        const auto& trigTrack = (*trackTriggers)[iTrigTrack];
        if (trigTrack.getBCData().differenceInBC(trig.getBCData()) > 0) {
          // aborting, since track trigger is later than digit trigger";
          break;
        }
        if (trigTrack.getBCData() != trig.getBCData()) {
          // skipping, since track trigger earlier than digit trigger";
          ++lastTrkTrig[iTrackType];
          continue;
        }
        if ((*tracks)[trigTrack.getFirstTrack()].hasPileUpInfo() && (*tracks)[trigTrack.getFirstTrack()].getPileUpTimeShiftMUS() < mParams.pileupCut) {
          // rejecting triggers which are close to other collisions (avoid pile-up)
          ++lastTrkTrig[iTrackType];
          break;
        }
        for (int iTrk = trigTrack.getFirstTrack(); iTrk < trigTrack.getFirstTrack() + trigTrack.getNumberOfTracks(); iTrk++) {
          const auto& trk = (*tracks)[iTrk];
          for (int iLayer = 0; iLayer < 6; iLayer++) {
            int trkltIdx = trk.getTrackletIndex(iLayer);
            if (trkltIdx < 0) {
              continue;
            }
            findDigitsForTracklet(mTrackletsRaw[trkltIdx], trig, iTrackType);
          }
        }
      }
    }
  }
  if (mWriteOutput) {
    mOutTree->Fill();
  }
}

void PulseHeight::findDigitsForTracklet(const Tracklet64& trklt, const TriggerRecord& trig, int type)
{
  auto trkltDet = trklt.getDetector();
  for (int iDigit = trig.getFirstDigit() + 1; iDigit < trig.getFirstDigit() + trig.getNumberOfDigits() - 1; ++iDigit) {
    const auto& digit = (*mDigits)[iDigit];
    if (digit.getDetector() != trkltDet || digit.getPadRow() != trklt.getPadRow() || digit.getPadCol() != trklt.getPadCol()) {
      // for now we loose charge information from padrow-crossing tracklets (~15% of all tracklets)
      continue;
    }
    int nNeighbours = 0;
    bool left = false;
    bool right = false;
    const auto& digitLeft = (*mDigits)[iDigit - 1];
    const auto& digitRight = (*mDigits)[iDigit + 1];
    LOG(debug) << "Central digit: " << digit;
    LOG(debug) << "Left digit: " << digitLeft;
    LOG(debug) << "Right digit: " << digitRight;
    if (digitLeft.isNeighbour(digit) && digitLeft.getChannel() < digit.getChannel()) {
      ++nNeighbours;
      left = true;
    }
    if (digitRight.isNeighbour(digit) && digitRight.getChannel() > digit.getChannel()) {
      ++nNeighbours;
      right = true;
    }
    if (nNeighbours > 0) {
      int digitTrackletDistance = 0;
      auto adcSumMax = digit.getADCsum();
      if (left && digitLeft.getADCsum() > adcSumMax) {
        adcSumMax = digitLeft.getADCsum();
        digitTrackletDistance = 1;
      }
      if (right && digitRight.getADCsum() > adcSumMax) {
        adcSumMax = digitRight.getADCsum();
        digitTrackletDistance = -1;
      }
      mDistances.push_back(digitTrackletDistance);
      for (int iTb = 0; iTb < TIMEBINS; ++iTb) {
        uint16_t phVal = digit.getADC()[iTb];
        if (left) {
          phVal += digitLeft.getADC()[iTb];
        }
        if (right) {
          phVal += digitRight.getADC()[iTb];
        }
        mPHValues.emplace_back(phVal, trkltDet, iTb, nNeighbours, type);
      }
    }
  }
}
