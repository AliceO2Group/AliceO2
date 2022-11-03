// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVSimulation/Digitizer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CPVBase/CPVSimParams.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/BasicCCDBManager.h"

#include <TRandom.h>
#include <fairlogger/Logger.h> // for LOG

ClassImp(o2::cpv::Digitizer);

using o2::cpv::Digit;
using o2::cpv::Hit;

using namespace o2::cpv;

//_______________________________________________________________________
void Digitizer::init()
{
  LOG(info) << "CPVDigitizer::init() : CCDB Url = " << o2::base::NameConf::getCCDBServer();
  if (o2::base::NameConf::getCCDBServer().compare("localtest") == 0) {
    mCalibParams = new CalibParams(1); // test default calibration
    mPedestals = new Pedestals(1);     // test default pedestals
    mBadMap = new BadChannelMap(1);    // test default bad channels
    LOG(info) << "[CPVDigitizer] No reading calibration from ccdb requested, set default";
  } else {
    auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
    ccdbMgr.setCaching(true);                     //make local cache of remote objects
    ccdbMgr.setLocalObjectValidityChecking(true); //query objects from remote site only when local one is not valid
    // read calibration from ccdb (for now do it only at the beginning of dataprocessing)
    // TODO: setup timestam according to anchors
    // Do not set timestamp here: This should be set from the framework and is done via the digitizer workflow
    // ccdbMgr.setTimestamp(o2::ccdb::getCurrentTimestamp());

    LOG(info) << "CCDB: Reading o2::cpv::CalibParams from CPV/Calib/Gains";
    mCalibParams = ccdbMgr.get<o2::cpv::CalibParams>("CPV/Calib/Gains");
    if (!mCalibParams) {
      LOG(error) << "Cannot get o2::cpv::CalibParams from CCDB. using dummy calibration!";
      mCalibParams = new CalibParams(1);
    }

    LOG(info) << "CCDB: Reading o2::cpv::Pedestals from CPV/Calib/Pedestals";
    mPedestals = ccdbMgr.get<o2::cpv::Pedestals>("CPV/Calib/Pedestals");
    if (!mPedestals) {
      LOG(error) << "Cannot get o2::cpv::Pedestals from CCDB. using dummy calibration!";
      mPedestals = new Pedestals(1);
    }

    LOG(info) << "CCDB: Reading o2::cpv::BadChannelMap from CPV/Calib/BadChannelMap";
    mBadMap = ccdbMgr.get<o2::cpv::BadChannelMap>("CPV/Calib/BadChannelMap");
    if (!mBadMap) {
      LOG(error) << "Cannot get o2::cpv::BadChannelMap from CCDB. using dummy calibration!";
      mBadMap = new BadChannelMap(1);
    }

    LOG(info) << "Task configuration is done.";
  }

  //signal thresolds for digits
  //note that digits are calibrated objects
  for (int i = 0; i < NCHANNELS; i++) {
    mDigitThresholds[i] = o2::cpv::CPVSimParams::Instance().mZSnSigmas *
                          mPedestals->getPedSigma(i) * mCalibParams->getGain(i);
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
  // Despite sorting in Detector::EndEvent(), hits still can be unsorted due to splitting of processing different bunches of primary
  for (int i = NCHANNELS; i--;) {
    mArrayD[i].reset();
  }

  // First, add pedestal noise and BG digits
  if (digitsBg.size() == 0) { // no digits provided: try simulate pedestal noise (do it only once)
    for (int i = NCHANNELS; i--;) {
      float amplitude = simulatePedestalNoise(i);
      if (amplitude > mDigitThresholds[i]) { //add noise digit if its signal > threshold
        mArrayD[i].setAmplitude(simulatePedestalNoise(i));
        mArrayD[i].setAbsId(i);
        // mArrayD[i].setLabel(-1); // noise marking (not needed to set as all mArrayD[i] elements are just resetted)
      }
    }
  } else {                       //if digits exist, noise is already added
    for (auto& dBg : digitsBg) { //digits are sorted and unique
      mArrayD[dBg.getAbsId()] = dBg;
    }
  }

  //Second, add Hits
  for (auto& h : *hits) {
    int i = h.GetDetectorID();
    if (mArrayD[i].getAmplitude() > 0) {
      mArrayD[i].setAmplitude(mArrayD[i].getAmplitude() + h.GetEnergyLoss()); //if amplitude > 0 then pedestal noise is already added
    } else {
      mArrayD[i].setAbsId(i);
      mArrayD[i].setAmplitude(h.GetEnergyLoss() + simulatePedestalNoise(i)); //if not then add pedestal noise to signal
    }
    if (mArrayD[i].getAmplitude() > mDigitThresholds[i]) {
      int labelIndex = mArrayD[i].getLabel();
      if (labelIndex == -1) { // noise
        labelIndex = labels.getIndexedSize();
        o2::MCCompLabel label(true); // noise label
        labels.addElement(labelIndex, label);
        mArrayD[i].setLabel(labelIndex);
      } else { //check if lable already exist
        gsl::span<MCCompLabel> sp = labels.getLabels(labelIndex);
        bool found = false;
        for (MCCompLabel& te : sp) {
          if (te.getTrackID() == h.GetTrackID() && te.getEventID() == collId && te.getSourceID() == source) {
            found = true;
            break;
          }
        }
        if (!found) {
          o2::MCCompLabel label(h.GetTrackID(), collId, source);
          //Highly inefficient management of Labels: commenting  line below reeduces WHOLE digitization time by factor ~30
          labels.addElementRandomAccess(labelIndex, label);
        }
      }
    }
  }

  //finalize output digits
  for (int i = 0; i < NCHANNELS; i++) {
    if (!mBadMap->isChannelGood(i)) {
      continue; //bad channel -> skip this digit
    }
    if (mArrayD[i].getAmplitude() > mDigitThresholds[i]) {
      digitsOut.push_back(mArrayD[i]);
    }
  }
}

float Digitizer::simulatePedestalNoise(int absId)
{
  //this function is to simulate pedestal and its noise (ADC counts)
  if (absId < 0 || absId >= NCHANNELS) {
    return 0.;
  }
  return gRandom->Gaus(0, mPedestals->getPedSigma(absId) * mCalibParams->getGain(absId));
}
