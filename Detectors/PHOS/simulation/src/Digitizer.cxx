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

#include <TRandom.h>
#include "FairLogger.h" // for LOG

ClassImp(o2::phos::Digitizer);

using o2::phos::Digit;
using o2::phos::Hit;

using namespace o2::phos;

//_______________________________________________________________________
void Digitizer::init() { mGeometry = Geometry::GetInstance(); }

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>& hits, std::vector<Digit>& digits)
{
  // Convert list of hits to digits:
  // Add hits with energy deposition in same cell and same time
  // Add energy corrections
  // Apply time smearing

  // Sort Hits: moved to Detector::FinishEvent
  // Add duplicates if any and remove them
  // TODO: Apply Poisson smearing of light production

  // std::vector<Hit> hits(ahits);

  // auto first = hits.begin();
  // auto last = hits.end();

  // std::sort(first, last);

  // first = hits.begin();
  // last = hits.end();

  // // this is copy of std::unique() method with addition: adding identical Hits
  // auto itr = first;
  // while (++first != last) {
  //   if (*itr == *first) {
  //     *itr += *first;
  //   } else {
  //     *(++itr) = *first;
  //   }
  // }
  // ++itr;

  // hits.erase(itr, hits.end());
  // // TODO==========End of hit sorting, to be moved to Detector=============

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
        hitAbsId = 99999; // out of PHOS
      }

      // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
      Double_t energy = digit.getAmplitude();
      // // Simulate electronic noise
      // energy += SimulateNoiseEnergy();

      // if (mApplyNonLinearity) {
      //   energy = NonLinearity(energy);
      // }
      // if (mApplyDigitization) {
      //   energy = DigitizeEnergy(energy);
      // }
      // if (mApplyDecalibration) {
      //   energy = Decalibrate(energy);
      // }
      // digit.setAmplitude(energy);
      if (mApplyTimeResolution) {
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
      if (energy > mZSthreshold) {
        Digit noiseDigit(absId, energy, time, -1); // current AbsId, energy, random time, no primary
        digits.push_back(noiseDigit);
      }
    }
  }
}

//_______________________________________________________________________
Double_t Digitizer::NonLinearity(const Double_t e) { return e * mcNL * (1. + maNL * exp(-e * e / 2. / mbNL / mbNL)); }
//_______________________________________________________________________
Double_t Digitizer::DigitizeEnergy(const Double_t e)
{
  // distretize energy if necessary
  return mADCWidth * ceil(e / mADCWidth);
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
  Double_t timeResolution = mTimeResolutionA + mTimeResolutionB / std::max(e, mTimeResThreshold);
  return gRandom->Gaus(time, timeResolution);
}
//_______________________________________________________________________
Double_t Digitizer::SimulateNoiseEnergy(void) { return DigitizeEnergy(gRandom->Gaus(0., mAPDNoise)); }
//_______________________________________________________________________
Double_t Digitizer::SimulateNoiseTime(void) { return gRandom->Uniform(kMinNoiseTime, kMaxNoiseTime); }

//_______________________________________________________________________
void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns
  t *= mCoeffToNanoSecond;

  if (t < mEventTime && mContinuous) {
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")" << FairLogger::endl;
  }
  mEventTime = t;
}

//_______________________________________________________________________
void Digitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID()
               << FairLogger::endl;
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void Digitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID()
               << FairLogger::endl;
  }
  mCurrEvID = v;
}
