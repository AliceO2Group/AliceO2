// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSSimulation/Digitizer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "PHOSSimulation/PHOSSimParams.h"

#include <TRandom.h>
#include "FairLogger.h" // for LOG

ClassImp(o2::phos::Digitizer);

using o2::phos::Digit;
using o2::phos::Hit;
using o2::phos::MCLabel;

using namespace o2::phos;

//_______________________________________________________________________
void Digitizer::init() { mGeometry = Geometry::GetInstance(); }

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>& hits, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::phos::MCLabel>& labels)
{
  // Convert list of hits to digits:
  // Add hits with energy deposition in same cell and same time
  // Add energy corrections
  // Apply time smearing

  Int_t hitIndex = 0;
  Int_t hitAbsId = 0;
  Int_t nHits = hits.size();
  Hit hit;
  if (hitIndex < nHits) {
    hit = hits.at(hitIndex);
    hitAbsId = hit.GetDetectorID();
  }

  Int_t nTotCells = mGeometry->GetTotalNCells();
  for (Int_t absId = 1; absId < nTotCells; absId++) {

    // If signal exist in this cell, add noise to it, otherwise just create noise digit
    if (absId == hitAbsId) {
      int labelIndex = labels.getIndexedSize();
      //Add primary info: create new MCLabels entry
      o2::phos::MCLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID, true, hit.GetEnergyLoss());
      labels.addElement(labelIndex, label);

      Digit digit(hit, labelIndex);

      hitIndex++;
      if (hitIndex < nHits) {
        Hit hitNext = hits.at(hitIndex);
        Digit digitNext(hitNext, -1); //Do not create MCTruth entry so far
        while ((hitIndex < nHits) && digit.canAdd(digitNext)) {
          digit += digitNext;

          //add MCLabel to list (add energy if same primary or add another label)
          o2::phos::MCLabel label(hitNext.GetTrackID(), mCurrEvID, mCurrSrcID, true, hitNext.GetEnergyLoss());
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
        hitAbsId = 99999; // out of PHOS
      }
      //Current digit finished, sort MCLabels according to eDeposited
      auto lbls = labels.getLabels(labelIndex);
      std::sort(lbls.begin(), lbls.end(),
                [](o2::phos::MCLabel a, o2::phos::MCLabel b) { return a.getEdep() > b.getEdep(); });

      // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
      Double_t energy = digit.getAmplitude();
      // Simulate electronic noise
      energy += SimulateNoiseEnergy();

      if (o2::phos::PHOSSimParams::Instance().mApplyNonLinearity) {
        energy = NonLinearity(energy);
      }
      if (o2::phos::PHOSSimParams::Instance().mApplyDigitization) {
        energy = DigitizeEnergy(energy);
      }
      if (o2::phos::PHOSSimParams::Instance().mApplyDecalibration) {
        energy = Decalibrate(energy);
      }
      digit.setAmplitude(energy);
      if (o2::phos::PHOSSimParams::Instance().mApplyTimeResolution) {
        digit.setTimeStamp(TimeResolution(digit.getTimeStamp(), energy));
      }
      digits.push_back(digit);
    } else { // No signal in this cell,
      if (!mGeometry->IsCellExists(absId)) {
        continue;
      }
      // Simulate noise
      Double_t energy = SimulateNoiseEnergy();
      Double_t time = SimulateNoiseTime();
      if (energy > o2::phos::PHOSSimParams::Instance().mZSthreshold) {
        Digit noiseDigit(absId, energy, time, -1); // current AbsId, energy, random time, no primary
        digits.push_back(noiseDigit);
      }
    }
  }
}

//_______________________________________________________________________
Double_t Digitizer::NonLinearity(const Double_t e)
{
  double a = o2::phos::PHOSSimParams::Instance().mCellNonLineaityA;
  double b = o2::phos::PHOSSimParams::Instance().mCellNonLineaityB;
  double c = o2::phos::PHOSSimParams::Instance().mCellNonLineaityC;
  return e * c * (1. + a * exp(-e * e / 2. / b / b));
}
//_______________________________________________________________________
Double_t Digitizer::DigitizeEnergy(const Double_t e)
{
  // distretize energy if necessary
  double w = o2::phos::PHOSSimParams::Instance().mADCwidth;
  return w * ceil(e / w);
}
//_______________________________________________________________________
Double_t Digitizer::Decalibrate(const Double_t e)
{
  // Decalibrate EMC digit, i.e. change its energy by a factor read from CDB
  return e; // TODO apply de-calibration from OCDB
}
//_______________________________________________________________________
Double_t Digitizer::TimeResolution(const Double_t time, const Double_t e)
{
  // apply time resolution

  Double_t timeResolution = o2::phos::PHOSSimParams::Instance().mTimeResolutionA +
                            o2::phos::PHOSSimParams::Instance().mTimeResolutionB /
                              std::max(float(e), o2::phos::PHOSSimParams::Instance().mTimeResThreshold);
  return gRandom->Gaus(time, timeResolution);
}
//_______________________________________________________________________
Double_t Digitizer::SimulateNoiseEnergy() { return DigitizeEnergy(gRandom->Gaus(0., o2::phos::PHOSSimParams::Instance().mAPDNoise)); }
//_______________________________________________________________________
Double_t Digitizer::SimulateNoiseTime() { return gRandom->Uniform(o2::phos::PHOSSimParams::Instance().mMinNoiseTime,
                                                                  o2::phos::PHOSSimParams::Instance().mMaxNoiseTime); }

//_______________________________________________________________________
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
