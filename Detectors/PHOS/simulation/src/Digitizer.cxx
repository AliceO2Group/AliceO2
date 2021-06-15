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
  if (!mTrigUtils) {
    if (o2::phos::PHOSSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mTrigUtils.reset(new TriggerMap(0)); // test default calibration
      LOG(INFO) << "[PHOSDigitizer] No reading trigger map from ccdb requested, set default";
    } else {
      LOG(ERROR) << "[PHOSDigitizer] can not get trigger map object from ccdb yet";
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
    float time = h.GetTime() + dt * 1.e-9;
    if (o2::phos::PHOSSimParams::Instance().mApplyTimeResolution) {
      time = uncalibrateT(timeResolution(time, energy), absId);
    }
    energy = uncalibrate(energy, absId);
    if (mArrayD[i].getAmplitude() > 0) {
      //update energy and time
      if (mArrayD[i].isHighGain()) {
        mArrayD[i].addEnergyTime(energy, time);
        //if overflow occured?
        if (mArrayD[i].getAmplitude() > o2::phos::PHOSSimParams::Instance().mMCOverflow) { //10bit ADC
          float hglgratio = mCalibParams->getHGLGRatio(absId);
          mArrayD[i].setAmplitude(mArrayD[i].getAmplitude() / hglgratio);
          mArrayD[i].setHighGain(false);
        }
      } else { //digit already in LG
        float hglgratio = mCalibParams->getHGLGRatio(absId);
        energy /= hglgratio;
        mArrayD[i].addEnergyTime(energy, time);
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
  //Calculate trigger tiles 2*2 and 4*4
  bool mL0Fired = false;
  const int nDDL = 14;
  const int nxTRU = 8;
  const int nzTRU = 28;
  float sum2x2[nxTRU + 1][nzTRU + 1];
  float time2x2[nxTRU + 1][nzTRU + 1];
  float tt = 0;
  for (char iTRU = 0; iTRU < nDDL; iTRU++) {
    for (char ix = 1; ix <= nxTRU; ix++) {
      for (char iz = 1; iz <= nzTRU; iz++) {
        char truRelId[3] = {iTRU, ix, iz};
        short tileId = Geometry::truRelToAbsNumbering(truRelId);
        if (!mTrigUtils->isGood2x2(tileId)) {
          continue;
        }
        char relId[3];
        Geometry::truRelId2RelId(truRelId, relId);
        short i1, i2, i3, i4;
        Geometry::relToAbsNumbering(relId, i1);
        relId[1] = relId[1] + 1;
        Geometry::relToAbsNumbering(relId, i2);
        relId[2] = relId[2] + 1;
        Geometry::relToAbsNumbering(relId, i4);
        relId[1] = relId[1] - 1;
        Geometry::relToAbsNumbering(relId, i3);
        sum2x2[ix][iz] = mArrayD[i1 - OFFSET].getAmplitude() + mArrayD[i2 - OFFSET].getAmplitude() +
                         mArrayD[i3 - OFFSET].getAmplitude() + mArrayD[i4 - OFFSET].getAmplitude();
        float ampMax = mArrayD[i1 - OFFSET].getAmplitude();
        tt = mArrayD[i1 - OFFSET].getTime();
        if (mArrayD[i2 - OFFSET].getAmplitude() > ampMax) {
          ampMax = mArrayD[i2 - OFFSET].getAmplitude();
          tt = mArrayD[i2 - OFFSET].getTime();
        }
        if (mArrayD[i3 - OFFSET].getAmplitude() > ampMax) {
          ampMax = mArrayD[i3 - OFFSET].getAmplitude();
          tt = mArrayD[i3 - OFFSET].getTime();
        }
        if (mArrayD[i4 - OFFSET].getAmplitude() > ampMax) {
          tt = mArrayD[i4 - OFFSET].getTime();
        }
        time2x2[ix][iz] = tt;
        if (mTrig2x2) {
          if (sum2x2[ix][iz] > PHOSSimParams::Instance().mTrig2x2MinThreshold) { //do not test (slow) probability function with soft tiles
            mL0Fired |= mTrigUtils->isFiredMC2x2(sum2x2[ix][iz], iTRU, short(ix), short(iz));
            //add TRU digit. Note that only tiles with E>mTrigMinThreshold added!
            digitsOut.emplace_back(tileId, sum2x2[ix][iz], tt, true, -1);
          }
        }
      }
    }

    if (mTrig4x4) {
      for (char ix = 1; ix < nxTRU; ix++) {
        for (char iz = 1; iz < nzTRU; iz++) {
          char truRelId[3] = {iTRU, ix, iz};
          short tileId = Geometry::truRelToAbsNumbering(truRelId);
          if (!mTrigUtils->isGood4x4(tileId)) {
            continue;
          }
          float sum4x4 = sum2x2[ix][iz] + sum2x2[ix][iz + 1] + sum2x2[ix + 1][iz] + sum2x2[ix + 1][iz + 1];
          if (sum4x4 > PHOSSimParams::Instance().mTrig4x4MinThreshold) { //do not test (slow) probability function with soft tiles
            mL0Fired |= mTrigUtils->isFiredMC4x4(sum4x4, iTRU, short(ix), short(iz));
            //Add TRU digit short cell, float amplitude, float time, int label
            tt = time2x2[ix][iz];
            float ampMax = sum2x2[ix][iz];
            if (sum2x2[ix][iz + 1] > ampMax) {
              ampMax = sum2x2[ix][iz + 1];
              tt = sum2x2[ix][iz + 1];
            }
            if (sum2x2[ix + 1][iz] > ampMax) {
              ampMax = sum2x2[ix + 1][iz];
              tt = sum2x2[ix + 1][iz];
            }
            if (sum2x2[ix + 1][iz + 1] > ampMax) {
              tt = sum2x2[ix][iz + 1];
            }
            digitsOut.emplace_back(tileId, sum4x4, tt, false, -1);
          }
        }
      }
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
