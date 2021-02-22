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
      mCalibParams.reset(new CalibParams(1)); // test default calibration
      LOG(INFO) << "[CPVDigitizer] No reading calibration from ccdb requested, set default";
    } else {
      LOG(INFO) << "[CPVDigitizer] can not gey calibration object from ccdb yet";
      //      o2::ccdb::CcdbApi ccdb;
      //      std::map<std::string, std::string> metadata; // do we want to store any meta data?
      //      ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
      //      mCalibParams = ccdb.retrieveFromTFileAny<o2::cpv::CalibParams>("CPV/Calib", metadata, mEventTime);
      //      if (!mCalibParams) {
      //        LOG(FATAL) << "[CPVDigitizer] can not get calibration object from ccdb";
      //      }
    }
  }
}

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::processHits(const std::vector<Hit>* hits, const std::vector<Digit>& digitsBg,
                            std::vector<Digit>& digitsOut, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels,
                            int collId, int source, double dt)
{
  // Convert list of hits to digits:
  // Add hits with ampl deposition in same pad and same time
  // Add ampl corrections
  // Apply time smearing

  std::vector<Digit>::const_iterator dBg = digitsBg.cbegin();
  std::vector<Hit>::const_iterator hBg = hits->cbegin();

  bool addNoise = !(digitsBg.size()); //If digits list not empty, noise already there

  int currentId = 0; //first digit

  while (dBg != digitsBg.cend() && hBg != hits->cend()) {
    if (dBg->getAbsId() < hBg->GetDetectorID()) { // copy digit
      //Digits already contain noise, no need to add it in this branch
      //Digits have correct time, no need to add event time
      digitsOut.emplace_back(dBg->getAbsId(), dBg->getAmplitude(), dBg->getLabel());
      currentId = dBg->getAbsId();
      dBg++;
    } else {
      if (addNoise) {
        addNoisyChannels(currentId, hBg->GetDetectorID(), digitsOut);
      }
      currentId = hBg->GetDetectorID();
      int labelIndex = -1;
      if (dBg->getAbsId() == hBg->GetDetectorID()) {
        labelIndex = dBg->getLabel();
      }
      if (labelIndex == -1) { //no digit or noisy
        labelIndex = labels.getIndexedSize();
      }
      Digit digit(*hBg, labelIndex);
      // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
      float energy = hBg->GetEnergyLoss();
      // Simulate electronic noise
      short absId = hBg->GetDetectorID();
      energy += simulateNoise();

      energy = uncalibrate(energy, absId);

      // Merge with existing digit if any
      if (dBg->getAbsId() == hBg->GetDetectorID()) {
        digit += *dBg;
        dBg++;
      }

      hBg++;
      if (energy <= o2::cpv::CPVSimParams::Instance().mZSthreshold) {
        continue;
      }

      //Add primary info: create new MCLabels entry
      o2::MCCompLabel label(hBg->GetTrackID(), collId, source, true);
      labels.addElementRandomAccess(labelIndex, label);

      digitsOut.push_back(digit);
    }
  }

  //Fill remainder
  while (dBg != digitsBg.cend()) {
    digitsOut.push_back(*dBg);
    dBg++;
  }

  while (hBg != hits->cend()) {
    if (addNoise) {
      addNoisyChannels(currentId, hBg->GetDetectorID(), digitsOut);
    }
    currentId = hBg->GetDetectorID();
    int labelIndex = labels.getIndexedSize();

    Digit digit(*hBg, labelIndex);
    // Add Electroinc noise
    float energy = hBg->GetEnergyLoss();
    // Simulate electronic noise
    short absId = hBg->GetDetectorID();
    energy += simulateNoise();
    energy = uncalibrate(energy, absId);
    hBg++;
    if (energy <= o2::cpv::CPVSimParams::Instance().mZSthreshold) {
      continue;
    }

    //Add primary info: create new MCLabels entry
    o2::MCCompLabel label(hBg->GetTrackID(), collId, source, true);
    labels.addElement(labelIndex, label);

    digitsOut.push_back(digit);
  }

  //Add noisy channels to the end of CPV
  if (addNoise) {
    addNoisyChannels(currentId, 128 * 60 * 3, digitsOut);
  }
}

float Digitizer::simulateNoise() { return gRandom->Gaus(0., o2::cpv::CPVSimParams::Instance().mNoise); }

//_______________________________________________________________________
void Digitizer::addNoisyChannels(int start, int end, std::vector<Digit>& digitsOut)
{

  // Simulate noise
  for (int absId = start + 1; absId < end; absId++) {
    float energy = simulateNoise();
    energy = uncalibrate(energy, absId);
    if (energy > o2::cpv::CPVSimParams::Instance().mZSthreshold) {
      digitsOut.emplace_back(absId, energy, -1); // current AbsId, energy, random time, no primary
    }
  }
}

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
