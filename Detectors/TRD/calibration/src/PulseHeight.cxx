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
}

void PulseHeight::init()
{
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
  LOGP(info, "Processing {} tracklets and {} digits from {} triggers and {} ITS-TPC-TRD tracks and {} TPC-TRD tracks",
       mTrackletsRaw.size(), mDigits->size(), mTriggerRecords.size(), mTracksInITSTPCTRD.size(), mTracksInTPCTRD.size());

  for (const auto& trigger : mTriggerRecords) {
    // skip triggers without digits
    if (trigger.getNumberOfDigits() == 0) {
      continue;
    }
    LOGP(info, "Trigger has {} tracklets and  {} digits", trigger.getNumberOfTracklets(), trigger.getNumberOfDigits());
    for (const auto& triggerTracks : mTrackTriggerRecordsITSTPCTRD) {
      // check if the bunch crossing matches for the trigger and the matched track
      if (trigger.getBCData() != triggerTracks.getBCData()) {
        continue;
      }

      int start = triggerTracks.getFirstTrack();
      int end = triggerTracks.getNumberOfTracks() + start;

      // check if the trigger track ends up in the pileup
      if (mTracksInITSTPCTRD[start].hasPileUpInfo()) {
        float pileUp = mTracksInITSTPCTRD[start].getPileUpTimeShiftMUS();
        LOGP(info, "Track {} has pileup of {} mus", start, pileUp);
      }

      for (int iTrack = start; iTrack < end; ++iTrack) {
        const auto& trkMatched = mTracksInITSTPCTRD[iTrack];

        for (int iLayer = 0; iLayer < NLAYER; ++iLayer) {
          // skipping tracks without tracklets
          if (trkMatched.getTrackletIndex(iLayer) < 0) {
            continue;
          }
          const auto& tracklet = mTrackletsRaw[trkMatched.getTrackletIndex(iLayer)];

          int trkltDet = tracklet.getDetector();
          int trkltPR = tracklet.getPadRow();
          int trkltPC = tracklet.getPadCol();

          // loop over digits where +1 and -1 are used to find consecutive digits
          for (int iDigit = trigger.getFirstDigit() + 1; iDigit < trigger.getFirstDigit() + trigger.getNumberOfDigits() - 1; ++iDigit) {
            const auto& digit = (*mDigits)[iDigit];
            const auto& digitPrev = (*mDigits)[iDigit - 1];
            const auto& digitNext = (*mDigits)[iDigit + 1];
            // check if the digit is in the same detector, pad row and pad column as the tracklet
            if (digit.getDetector() != trkltDet || digit.getPadRow() != trkltPR || (TMath::Abs(digit.getPadCol() - trkltPC) > 1)) {
              continue;
            }
            // check for consecutive digits
            if (digit.getDetector() == digitPrev.getDetector() && digit.getPadRow() == digitPrev.getPadRow() && digit.getPadRow() == digitNext.getPadRow() && digit.getPadCol() == digitPrev.getPadCol() - 1 && digit.getPadCol() == digitNext.getPadCol() + 1) {
              // calculate the pulse height
              for (int iTBin = 0; iTBin < TIMEBINS; ++iTBin) {
                float adcVal = digitPrev.getADC()[iTBin] + digit.getADC()[iTBin] + digitNext.getADC()[iTBin];
                LOGP(info, "ADC value is {}", adcVal);
              }
            }
          }
        }
      }
    }
  }
}
