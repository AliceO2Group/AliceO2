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
#include "CPVSimulation/CPVSimParams.h"

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
void Digitizer::process(const std::vector<Hit>& hits, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels)
{
  // Convert list of hits to digits:
  // Add hits with ampl deposition in same pad and same time
  // Add ampl corrections
  // Apply time smearing

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
      int labelIndex = labels.getIndexedSize();
      //Add primary info: create new MCLabels entry
      o2::MCCompLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID, true);
      labels.addElement(labelIndex, label);

      Digit digit(hit, labelIndex);

      hitIndex++;
      if (hitIndex < nHits) {
        Hit hitNext = hits.at(hitIndex);
        Digit digitNext(hitNext, -1); //Do not create MCTruth entry so far
        while ((hitIndex < nHits) && digit.canAdd(digitNext)) {
          digit += digitNext;

          //add MCLabel to list (add energy if same primary or add another label)
          o2::MCCompLabel label(hitNext.GetTrackID(), mCurrEvID, mCurrSrcID, true);
          labels.addElementRandomAccess(labelIndex, label);

          hitIndex++;
          if (hitIndex < nHits) {
            hitNext = hits.at(hitIndex);
            digitNext.FillFromHit(hitNext);
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
      //      //Current digit finished, sort MCLabels according to eDeposited
      //      auto lbls = labels.getLabels(labelIndex);
      //      std::sort(lbls.begin(), lbls.end(),
      //                [](o2::MCCompLabel a, o2::MCCompLabel b) { return a.getEdep() > b.getEdep(); });

      // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
      Double_t ampl = digit.getAmplitude();
      // Simulate electronic noise
      ampl += SimulateNoise();

      if (o2::cpv::CPVSimParams::Instance().mApplyDigitization) {
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
      if (ampl > o2::cpv::CPVSimParams::Instance().mZSthreshold) {
        if (o2::cpv::CPVSimParams::Instance().mApplyDigitization) {
          ampl = DigitizeAmpl(ampl);
        }
        Digit noiseDigit(absId, ampl, mEventTime, -1); // current AbsId, ampl, random time, no primary
        digits.push_back(noiseDigit);
      }
    }
  }
}

Double_t Digitizer::SimulateNoise() { return gRandom->Gaus(0., o2::cpv::CPVSimParams::Instance().mNoise); }

Double_t Digitizer::DigitizeAmpl(double a) { return o2::cpv::CPVSimParams::Instance().mADCWidth * TMath::Ceil(a / o2::cpv::CPVSimParams::Instance().mADCWidth); }

void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns
  t *= o2::cpv::CPVSimParams::Instance().mCoeffToNanoSecond;

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
