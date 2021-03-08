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
  // //Despite sorting in Detector::EndEvent(), hits still can be unsorted due to splitting of processing different bunches of primary
  for (int i = NCHANNELS; i--;) {
    mArrayD[i].reset();
  }

  if (digitsBg.size() == 0) { // no digits provided: try simulate noise
    for (int i = NCHANNELS; i--;) {
      float energy = simulateNoiseEnergy(i + OFFSET);
      energy = uncalibrate(energy, i + OFFSET);
      if (energy > o2::phos::PHOSSimParams::Instance().mDigitThreshold) {
        float time = simulateNoiseTime();
        mArrayD[i].setAmplitude(energy);
        mArrayD[i].setTime(time);
        mArrayD[i].setAbsId(i + OFFSET);
      }
    }
  } else {                       //if digits exist, no noise should be added
    for (auto& dBg : digitsBg) { //digits are sorted and unique
      mArrayD[dBg.getAbsId() - OFFSET] = dBg;
    }
  }

  //add Hits
  for (auto& h : *hits) {
    short absId = h.GetDetectorID();
    short i = absId - OFFSET;
    float energy = h.GetEnergyLoss();
    if (o2::phos::PHOSSimParams::Instance().mApplyNonLinearity) {
      energy = nonLinearity(energy);
    }
    energy = uncalibrate(energy, absId);
    float time = h.GetTime() + dt * 1.e-9;
    if (o2::phos::PHOSSimParams::Instance().mApplyTimeResolution) {
      time = uncalibrateT(timeResolution(time, energy), absId);
    }
    if (mArrayD[i].getAmplitude() > 0) {
      //update energy and time
      mArrayD[i].addEnergyTime(energy, time);
      //if overflow occured?
      if (mArrayD[i].isHighGain()) {
        if (mArrayD[i].getAmplitude() > o2::phos::PHOSSimParams::Instance().mMCOverflow) { //10bit ADC
          float hglgratio = mCalibParams->getHGLGRatio(absId);
          mArrayD[i].setAmplitude(mArrayD[i].getAmplitude() / hglgratio);
          mArrayD[i].setHighGain(false);
        }
      }
    } else {
      mArrayD[i].setHighGain(energy < o2::phos::PHOSSimParams::Instance().mMCOverflow); //10bit ADC
      if (mArrayD[i].isHighGain()) {
        mArrayD[i].setAmplitude(energy);
      } else {
        float hglgratio = mCalibParams->getHGLGRatio(absId);
        mArrayD[i].setAmplitude(energy / hglgratio);
      }
      mArrayD[i].setTime(time);
      mArrayD[i].setAbsId(absId);
    }
    //Add MC info
    if (mProcessMC) {
      int labelIndex = mArrayD[i].getLabel();
      if (labelIndex == -1) { //no digit or noisy
        labelIndex = labels.getIndexedSize();
        MCLabel label(h.GetTrackID(), collId, source, false, h.GetEnergyLoss());
        labels.addElement(labelIndex, label);
        mArrayD[i].setLabel(labelIndex);
      } else { //check if lable already exist
        MCLabel label(h.GetTrackID(), collId, source, false, h.GetEnergyLoss());
        gsl::span<MCLabel> sp = labels.getLabels(labelIndex);
        bool found = false;
        for (MCLabel& te : sp) {
          if (te == label) {
            found = true;
            te.add(label, 1.);
            break;
          }
        }
        if (!found) {
          //Highly inefficient management of Labels: commenting  line below reeduces WHOLE digitization time by factor ~30
          labels.addElementRandomAccess(labelIndex, label);
          //sort MCLabels according to eDeposited
          sp = labels.getLabels(labelIndex);
          std::sort(sp.begin(), sp.end(),
                    [](o2::phos::MCLabel a, o2::phos::MCLabel b) { return a.getEdep() > b.getEdep(); });
        }
      }
    } else {
      mArrayD[i].setLabel(-1);
    }
  }
  for (int i = 0; i < NCHANNELS; i++) {
    if (mArrayD[i].getAmplitude() > PHOSSimParams::Instance().mZSthreshold) {
      digitsOut.push_back(mArrayD[i]);
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
float Digitizer::uncalibrateT(const float time, const int absId)
{
  // Decalibrate EMC digit, i.e. transform from energy to ADC counts a factor read from CDB
  // note time in seconds
  return time + mCalibParams->getHGTimeCalib(absId);
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
