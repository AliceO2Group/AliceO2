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
#include "CPVBase/CPVSimParams.h"
#include "CCDB/CcdbApi.h"

#include <TRandom.h>
#include "FairLogger.h" // for LOG

ClassImp(o2::cpv::Digitizer);

using o2::cpv::Digit;
using o2::cpv::Hit;

using namespace o2::cpv;

//_______________________________________________________________________
void Digitizer::init()
{
  if (!mCalibParams) {
    if (o2::cpv::CPVSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mCalibParams = new CalibParams(1); // test default calibration
      LOG(INFO) << "[CPVDigitizer] No reading calibration from ccdb requested, set default";
    } else {
      LOG(INFO) << "[CPVDigitizer] getting calibration object from ccdb";
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata; // do we want to store any meta data?
      ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
      mCalibParams = ccdb.retrieveFromTFileAny<o2::cpv::CalibParams>("CPV/Calib", metadata, mEventTime);
      if (!mCalibParams) {
        LOG(FATAL) << "[CPVDigitizer] can not get calibration object from ccdb";
      }
    }
  }
}

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>* hitsBg, const std::vector<Hit>* hitsS, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels)
{
  // Convert list of hits to digits:
  // Add hits with ampl deposition in same pad and same time
  // Add ampl corrections
  // Apply time smearing

  std::vector<Hit>::const_iterator hitBg = hitsBg->cbegin();
  std::vector<Hit>::const_iterator hitS = hitsS->cbegin();
  std::vector<Hit>::const_iterator hit;
  const short kBigAbsID = 32767; //Far above maximal CPV absId
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

  short nTotPads = Geometry::getTotalNPads();
  for (short absId = 1; absId < nTotPads; absId++) {

    // If signal exist in this pad, add noise to it, otherwise just create noise digit
    if (absId == hitAbsId) {
      int labelIndex = labels.getIndexedSize();
      //Add primary info: create new MCLabels entry
      o2::MCCompLabel label(hit->GetTrackID(), mCurrEvID, mCurrSrcID, true);
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
        o2::MCCompLabel label(hit->GetTrackID(), mCurrEvID, mCurrSrcID, true);
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
      //      //Current digit finished, sort MCLabels according to eDeposited
      //      auto lbls = labels.getLabels(labelIndex);
      //      std::sort(lbls.begin(), lbls.end(),
      //                [](o2::MCCompLabel a, o2::MCCompLabel b) { return a.getEdep() > b.getEdep(); });

      // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
      float ampl = digit.getAmplitude();
      // Simulate electronic noise
      ampl += simulateNoise();

      ampl = uncalibrate(ampl, absId);

      if (ampl < o2::cpv::CPVSimParams::Instance().mZSthreshold) {
        continue;
      }
      digit.setAmplitude(ampl);

      digits.push_back(digit);
    } else { // No signal in this pad,
      if (!Geometry::IsPadExists(absId)) {
        continue;
      }
      // Simulate noise
      float ampl = simulateNoise();
      ampl = uncalibrate(ampl, absId);
      if (ampl > o2::cpv::CPVSimParams::Instance().mZSthreshold) {
        digits.emplace_back(absId, ampl, -1); // current AbsId, energy, no primary
      }
    }
  }
}

float Digitizer::simulateNoise() { return gRandom->Gaus(0., o2::cpv::CPVSimParams::Instance().mNoise); }

//_______________________________________________________________________
float Digitizer::uncalibrate(const float e, const int absId)
{
  // Decalibrate CPV digit, i.e. transform from amplitude to ADC counts a factor read from CDB
  float calib = mCalibParams->getGain(absId);
  if (calib > 0) {
    return floor(e / calib);
  } else {
    return 0; // TODO apply de-calibration from OCDB
  }
}

void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns

  if (t < mEventTime) {
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")";
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
