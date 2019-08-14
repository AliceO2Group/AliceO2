// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVSimulation/Digitizer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <TRandom.h>
#include "FairLogger.h" // for LOG

ClassImp(o2::cpv::Digitizer);

using o2::cpv::Digit;
using o2::cpv::Hit;

using namespace o2::cpv;

//_______________________________________________________________________
void Digitizer::init() { mGeometry = Geometry::GetInstance(); }

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>& hits, std::vector<Digit>& digits)
{
  // Convert list of hits to digits:
  // Add hits with ampl deposition in same pad and same time
  // Add ampl corrections
  // Apply time smearing

  // Sort Hits: moved to Detector::FinishEvent
  // Add duplicates if any and remove them

  Int_t hitIndex = 0;
  Int_t hitAbsId = 0;
  Int_t nHits = hits.size();
  Hit hit;
  if (hitIndex < nHits) {
    hit = hits.at(hitIndex);
    hitAbsId = hit.GetDetectorID();
  }

  Int_t nTotPads = mGeometry->GetTotalNPads();
  for (Int_t absId = 1; absId < nTotPads; absId++) {

    // If signal exist in this pad, add noise to it, otherwise just create noise digit
    if (absId == hitAbsId) {
      Digit digit(hit);
      hitIndex++;
      if (hitIndex < nHits) {
        Digit digitNext(hits.at(hitIndex));
        while ((hitIndex < nHits) && digit.canAdd(digitNext)) {
          digit += digitNext;
          hitIndex++;
          if (hitIndex < nHits) {
            digitNext.FillFromHit(hits.at(hitIndex));
          }
        }
        if (hitIndex < nHits) {
          hitAbsId = digitNext.getAbsId();
        } else {
          hitAbsId = 0;
        }
        hit = hits.at(hitIndex);
      } else {
        hitAbsId = 99999; // out of CPV
      }

      // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
      Double_t ampl = digit.getAmplitude();
      // Simulate electronic noise
      ampl += SimulateNoise();

      if (mApplyDigitization) {
        ampl = DigitizeAmpl(ampl);
      }
      digit.setAmplitude(ampl);
      digits.push_back(digit);
    } else { // No signal in this pad,
      if (!mGeometry->IsPadExists(absId)) {
        continue;
      }
      // Simulate noise
      Double_t ampl = SimulateNoise();
      if (ampl > mZSthreshold) {
        if (mApplyDigitization) {
          ampl = DigitizeAmpl(ampl);
        }
        Digit noiseDigit(absId, ampl, mEventTime, -1); // current AbsId, ampl, random time, no primary
        digits.push_back(noiseDigit);
      }
    }
  }
}

Double_t Digitizer::SimulateNoise() { return gRandom->Gaus(0., mNoise); }

Double_t Digitizer::DigitizeAmpl(double a) { return mADCWidth * TMath::Ceil(a / mADCWidth); }

void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns
  t *= mCoeffToNanoSecond;

  if (t < mEventTime && mContinuous) {
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")";
  }
  mEventTime = t;
}

//_______________________________________________________________________
void Digitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID();
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void Digitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID();
  }
  mCurrEvID = v;
}
