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
  if (!mCalibParams) {
    if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mCalibParams.reset(new CalibParams(1)); // test default calibration
      LOG(INFO) << "[PHOSDigitizer] No reading calibration from ccdb requested, set default";
    } else {
      LOG(ERROR) << "[PHOSDigitizer] can not get calibration object from ccdb yet";
      // o2::ccdb::CcdbApi ccdb;
      // std::map<std::string, std::string> metadata; // do we want to store any meta data?
      // ccdb.init("http://ccdb-test.cern.ch:8080");  // or http://localhost:8080 for a local installation
      // mCalibParams = ccdb.retrieveFromTFileAny<o2::phos::CalibParams>("PHOS/Calib", metadata, mEventTime);
      // if (!mCalibParams) {
      //   LOG(FATAL) << "[PHOSDigitizer] can not get calibration object from ccdb";
      // }
    }
  }
}

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::processHits(const std::vector<Hit>* hits, const std::vector<Digit>& digitsBg,
                            std::vector<Digit>& digitsOut, o2::dataformats::MCTruthContainer<MCLabel>& labels,
                            int collId, int source, double dt)
{
  // Convert list of hits + possible Bg digits to  digits:
  // Add hits with energy deposition in same cell and same time
  // Add energy corrections
  // Apply time smearing

  std::vector<Digit>::const_iterator dBg = digitsBg.cbegin();
  std::vector<Hit>::const_iterator hBg = hits->cbegin();

  bool addNoise = !(digitsBg.size()); //If digits list not empty, noise already there

  int currentId = 64 * 56 + 32 * 56; //first digit in half-mod 1 minus 1

  while (dBg != digitsBg.cend() && hBg != hits->cend()) {
    if (dBg->getAbsId() < hBg->GetDetectorID()) { // copy digit
      //Digits already contain noise, no need to add it in this branch
      //Digits have correct time, no need to add event time
      digitsOut.emplace_back(dBg->getAbsId(), dBg->getAmplitude(), dBg->getTime(), dBg->getLabel());
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
      energy += simulateNoiseEnergy(absId);

      if (o2::phos::PHOSSimParams::Instance().mApplyNonLinearity) {
        energy = nonLinearity(energy);
      }

      energy = uncalibrate(energy, absId);

      digit.setHighGain(energy < o2::phos::PHOSSimParams::Instance().mMCOverflow); //10bit ADC
      if (digit.isHighGain()) {
        digit.setAmplitude(energy);
      } else {
        float hglgratio = mCalibParams->getHGLGRatio(absId);
        digit.setAmplitude(energy / hglgratio);
      }

      float time = hBg->GetTime() + dt * 1.e-9;
      if (o2::phos::PHOSSimParams::Instance().mApplyTimeResolution) {
        digit.setTime(uncalibrateT(timeResolution(time, energy), absId, digit.isHighGain()));
      }

      // Merge with existing digit if any
      if (dBg->getAbsId() == hBg->GetDetectorID()) {
        digit += *dBg;
        dBg++;
      }

      hBg++;
      if (energy <= o2::phos::PHOSSimParams::Instance().mDigitThreshold) {
        continue;
      }

      //Add primary info: create new MCLabels entry
      o2::phos::MCLabel label(hBg->GetTrackID(), collId, source, true, hBg->GetEnergyLoss());
      labels.addElementRandomAccess(labelIndex, label);
      //sort MCLabels according to eDeposited
      auto lbls = labels.getLabels(labelIndex);
      std::sort(lbls.begin(), lbls.end(),
                [](o2::phos::MCLabel a, o2::phos::MCLabel b) { return a.getEdep() > b.getEdep(); });

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
    // Add Electroinc noise, apply non-linearity, digitize, de-calibrate, time resolution
    float energy = hBg->GetEnergyLoss();
    // Simulate electronic noise
    short absId = hBg->GetDetectorID();
    energy += simulateNoiseEnergy(absId);

    if (o2::phos::PHOSSimParams::Instance().mApplyNonLinearity) {
      energy = nonLinearity(energy);
    }

    energy = uncalibrate(energy, absId);

    digit.setHighGain(energy < o2::phos::PHOSSimParams::Instance().mMCOverflow); //10bit ADC
    if (digit.isHighGain()) {
      digit.setAmplitude(energy);
    } else {
      float hglgratio = mCalibParams->getHGLGRatio(absId);
      digit.setAmplitude(energy / hglgratio);
    }

    float time = hBg->GetTime() + dt * 1.e-9;
    if (o2::phos::PHOSSimParams::Instance().mApplyTimeResolution) {
      digit.setTime(uncalibrateT(timeResolution(time, energy), absId, digit.isHighGain()));
    }

    hBg++;
    if (energy <= o2::phos::PHOSSimParams::Instance().mDigitThreshold) {
      continue;
    }

    //Add primary info: create new MCLabels entry
    o2::phos::MCLabel label(hBg->GetTrackID(), collId, source, true, hBg->GetEnergyLoss());
    labels.addElement(labelIndex, label);

    digitsOut.push_back(digit);
  }

  //Add noisy channels to the end of PHOS
  if (addNoise) {
    addNoisyChannels(currentId, 56 * 64 * 4, digitsOut);
  }
}

//_______________________________________________________________________
void Digitizer::addNoisyChannels(int start, int end, std::vector<Digit>& digitsOut)
{

  // Simulate noise
  for (int absId = start + 1; absId < end; absId++) {
    float energy = simulateNoiseEnergy(absId);
    energy = uncalibrate(energy, absId);
    if (energy > o2::phos::PHOSSimParams::Instance().mDigitThreshold) {
      float time = simulateNoiseTime();
      digitsOut.emplace_back(absId, energy, time, -1); // current AbsId, energy, random time, no primary
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
    return 0;
  }
}
//_______________________________________________________________________
float Digitizer::uncalibrateT(const float time, const int absId, bool isHighGain)
{
  // Decalibrate EMC digit, i.e. transform from energy to ADC counts a factor read from CDB
  // note time in seconds
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
  // time measured in seconds

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
