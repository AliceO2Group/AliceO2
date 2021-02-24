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
  // //Despite sorting in Detector::EndEvent(), hits still can be unsorted due to splitting of processing different bunches of primary
  for (int i = NCHANNELS; i--;) {
    mArrayD[i].reset();
  }

  if (digitsBg.size() == 0) { // no digits provided: try simulate noise
    for (int i = NCHANNELS; i--;) {
      float energy = simulateNoise();
      energy = uncalibrate(energy, i);
      if (energy > o2::cpv::CPVSimParams::Instance().mZSthreshold) {
        mArrayD[i].setAmplitude(energy);
        mArrayD[i].setAbsId(i);
      }
    }
  } else {                       //if digits exist, no noise should be added
    for (auto& dBg : digitsBg) { //digits are sorted and unique
      mArrayD[dBg.getAbsId()] = dBg;
    }
  }

  //add Hits
  for (auto& h : *hits) {
    int i = h.GetDetectorID();
    if (mArrayD[i].getAmplitude() > 0) {
      mArrayD[i].setAmplitude(mArrayD[i].getAmplitude() + uncalibrate(h.GetEnergyLoss(), i));
    } else {
      mArrayD[i].setAmplitude(uncalibrate(h.GetEnergyLoss(), i));
      mArrayD[i].setAbsId(i);
    }
    if (mArrayD[i].getAmplitude() > o2::cpv::CPVSimParams::Instance().mZSthreshold) {
      int labelIndex = mArrayD[i].getLabel();
      if (labelIndex == -1) { //no digit or noisy
        labelIndex = labels.getIndexedSize();
        o2::MCCompLabel label(h.GetTrackID(), collId, source, true);
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
          o2::MCCompLabel label(h.GetTrackID(), collId, source, true);
          //Highly inefficient management of Labels: commenting  line below reeduces WHOLE digitization time by factor ~30
          labels.addElementRandomAccess(labelIndex, label);
        }
      }
    }
  }

  for (int i = 0; i < NCHANNELS; i++) {
    if (mArrayD[i].getAmplitude() > o2::cpv::CPVSimParams::Instance().mZSthreshold) {
      digitsOut.push_back(mArrayD[i]);
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
    return e / calib;
  } else {
    return 0; // TODO apply de-calibration from OCDB
  }
}
