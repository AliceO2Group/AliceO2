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
#include "PHOSBase/PHOSSimParams.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "CCDB/CcdbApi.h"

#include <TRandom.h>
#include "FairLogger.h" // for LOG

ClassImp(o2::phos::Digitizer);

using o2::phos::Hit;

using namespace o2::phos;

//_______________________________________________________________________
void Digitizer::init()
{
  mGeometry = Geometry::GetInstance();
  if (!mCalibParams) {
    if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mCalibParams = new CalibParams(1); // test default calibration
      LOG(INFO) << "[PHOSDigitizer] No reading calibration from ccdb requested, set default";
    } else {
      LOG(INFO) << "[PHOSDigitizer] getting calibration object from ccdb";
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata; // do we want to store any meta data?
      ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
      mCalibParams = ccdb.retrieveFromTFileAny<o2::phos::CalibParams>("PHOS/Calib", metadata, mEventTime);
      if (!mCalibParams) {
        LOG(FATAL) << "[PHOSDigitizer] can not get calibration object from ccdb";
      }
    }
  }
}

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>* hitsBg, const std::vector<Hit>* hitsS, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::phos::MCLabel>& labels)
{
  // Convert list of hits to digits:
  // Add hits with energy deposition in same cell and same time
  // Add energy corrections
  // Apply time smearing

  std::vector<Hit>::const_iterator hitBg = hitsBg->cbegin();
  std::vector<Hit>::const_iterator hitS = hitsS->cbegin();
  std::vector<Hit>::const_iterator hit; //Far above maximal PHOS absId
  const short kBigAbsID = 32767;
  short hitAbsId = kBigAbsID;
  short hitBgAbsId = kBigAbsID;
  short hitSAbsId = kBigAbsID;

  if (hitBg != hitsBg->end()) {
    hitBgAbsId = hitBg->GetDetectorID();
  }
  if (hitS != hitsS->end()) {
    hitSAbsId = hitS->GetDetectorID();
  }
  if (hitBgAbsId < hitSAbsId) { // Bg hit exists and smaller than signal
    hitAbsId = hitBgAbsId;
    hit = hitBg;
    mCurrSrcID = 0;
    ++hitBg;
  } else {
    if (hitSAbsId < kBigAbsID) { //Signal hit exists and smaller than Bg
      hitAbsId = hitSAbsId;
      hit = hitS;
      mCurrSrcID = 1;
      ++hitS;
    }
  }

  Int_t nTotCells = mGeometry->getTotalNCells();
  for (short absId = 1; absId < nTotCells; absId++) {

    // If signal exist in this cell, add noise to it, otherwise just create noise digit
    if (absId == hitAbsId) {
      int labelIndex = labels.getIndexedSize();
      //Add primary info: create new MCLabels entry
      o2::phos::MCLabel label(hit->GetTrackID(), mCurrEvID, mCurrSrcID, true, hit->GetEnergyLoss());
      labels.addElement(labelIndex, label);

      Digit digit((*hit), labelIndex);

      //May be add more hits to this digit?
      if (hitBg == hitsBg->end()) {
        hitBgAbsId = kBigAbsID;
      } else {
        hitBgAbsId = hitBg->GetDetectorID();
      }
      if (hitS == hitsS->end()) {
        hitSAbsId = kBigAbsID;
      } else {
        hitSAbsId = hitS->GetDetectorID();
      }
      if (hitBgAbsId < hitSAbsId) { // Bg hit exists and smaller than signal
        hitAbsId = hitBgAbsId;
        hit = hitBg;
        mCurrSrcID = 0;
        ++hitBg;
      } else {
        if (hitSAbsId < kBigAbsID) { //Signal hit exists and smaller than Bg
          hitAbsId = hitSAbsId;
          hit = hitS;
          mCurrSrcID = 1;
          ++hitS;
        } else { //no hits left
          hitAbsId = kBigAbsID;
          continue;
        }
      }

      while (absId == hitAbsId) {
        Digit digitNext((*hit), labelIndex); //Use same MCTruth entry so far
        digit += digitNext;

        //add MCLabel to list (add energy if same primary or add another label)
        o2::phos::MCLabel label(hit->GetTrackID(), mCurrEvID, mCurrSrcID, true, hit->GetEnergyLoss());
        labels.addElement(labelIndex, label);

        //next hit?
        if (hitBg == hitsBg->end()) {
          hitBgAbsId = kBigAbsID;
        } else {
          hitBgAbsId = hitBg->GetDetectorID();
        }
        if (hitS == hitsS->end()) {
          hitSAbsId = kBigAbsID;
        } else {
          hitSAbsId = hitS->GetDetectorID();
        }
        if (hitBgAbsId < hitSAbsId) { // Bg hit exists and smaller than signal
          hitAbsId = hitBgAbsId;
          hit = hitBg;
          mCurrSrcID = 0;
          ++hitBg;
        } else {
          if (hitSAbsId < kBigAbsID) { //Signal hit exists and smaller than Bg
            hitAbsId = hitSAbsId;
            hit = hitS;
            mCurrSrcID = 1;
            ++hitS;
          } else { //no hits left
            hitAbsId = kBigAbsID;
            digitNext.setAbsId(kBigAbsID);
            continue;
          }
        }

        digitNext.fillFromHit(*hit);
      }

      //Current digit finished, sort MCLabels according to eDeposited
      auto lbls = labels.getLabels(labelIndex);
      std::sort(lbls.begin(), lbls.end(),
                [](o2::phos::MCLabel a, o2::phos::MCLabel b) { return a.getEdep() > b.getEdep(); });

      // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
      float energy = digit.getAmplitude();
      // Simulate electronic noise
      energy += simulateNoiseEnergy(absId);

      if (o2::phos::PHOSSimParams::Instance().mApplyNonLinearity) {
        energy = nonLinearity(energy);
      }

      energy = uncalibrate(energy, absId);

      if (energy < o2::phos::PHOSSimParams::Instance().mZSthreshold) {
        continue;
      }
      digit.setAmplitude(energy);
      digit.setHighGain(energy < 1024); //10bit ADC

      if (o2::phos::PHOSSimParams::Instance().mApplyTimeResolution) {
        digit.setTimeStamp(uncalibrateT(timeResolution(digit.getTimeStamp(), energy), absId, digit.isHighGain()));
      }

      digits.push_back(digit);
    } else { // No signal in this cell,
      if (!mGeometry->isCellExists(absId)) {
        continue;
      }
      // Simulate noise
      float energy = simulateNoiseEnergy(absId);
      energy = uncalibrate(energy, absId);
      float time = simulateNoiseTime();
      if (energy > o2::phos::PHOSSimParams::Instance().mZSthreshold) {
        digits.emplace_back(absId, energy, time, -1); // current AbsId, energy, random time, no primary
      }
    }
  }
}

//_______________________________________________________________________
float Digitizer::nonLinearity(const float e)
{
  float a = o2::phos::PHOSSimParams::Instance().mCellNonLineaityA;
  float b = o2::phos::PHOSSimParams::Instance().mCellNonLineaityB;
  float c = o2::phos::PHOSSimParams::Instance().mCellNonLineaityC;
  return e * c * (1. + a * exp(-e * e / (2. * b * b)));
}
//_______________________________________________________________________
float Digitizer::uncalibrate(const float e, const int absId)
{
  // Decalibrate EMC digit, i.e. transform from energy to ADC counts a factor read from CDB
  float calib = mCalibParams->getGain(absId);
  if (calib > 0) {
    return floor(e / calib);
  } else {
    return 0; // TODO apply de-calibration from OCDB
  }
}
//_______________________________________________________________________
float Digitizer::uncalibrateT(const float time, const int absId, bool isHighGain)
{
  // Decalibrate EMC digit, i.e. transform from energy to ADC counts a factor read from CDB
  if (isHighGain) {
    return time + mCalibParams->getHGTimeCalib(absId);
  } else {
    return time + mCalibParams->getLGTimeCalib(absId);
  }
}
//_______________________________________________________________________
float Digitizer::timeResolution(const float time, const float e)
{
  // apply time resolution

  float timeResolution = o2::phos::PHOSSimParams::Instance().mTimeResolutionA +
                         o2::phos::PHOSSimParams::Instance().mTimeResolutionB /
                           std::max(float(e), o2::phos::PHOSSimParams::Instance().mTimeResThreshold);
  return gRandom->Gaus(time, timeResolution);
}
//_______________________________________________________________________
float Digitizer::simulateNoiseEnergy(int absId)
{
  return gRandom->Gaus(0., o2::phos::PHOSSimParams::Instance().mAPDNoise);
}
//_______________________________________________________________________
float Digitizer::simulateNoiseTime() { return gRandom->Uniform(o2::phos::PHOSSimParams::Instance().mMinNoiseTime,
                                                               o2::phos::PHOSSimParams::Instance().mMaxNoiseTime); }

//_______________________________________________________________________
void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // in ns

  if (t < mEventTime) {
    LOG(INFO) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")";
  }
  mEventTime = t;
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
