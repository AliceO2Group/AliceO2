// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALSimulation/Digitizer.h"
#include "EMCALSimulation/SimParam.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <climits>
#include <forward_list>
#include <chrono>
#include <TRandom.h>
#include <TF1.h>
#include "FairLogger.h" // for LOG

ClassImp(o2::emcal::Digitizer);

using o2::emcal::Digit;
using o2::emcal::Hit;

using namespace o2::emcal;

//_______________________________________________________________________
void Digitizer::init()
{
  mSimParam = SimParam::getInstance();
  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  float tau = mSimParam->getTimeResponseTau();
  float N = mSimParam->getTimeResponsePower();
  mTimeBinOffset = ((int)tau + 0.5);
  mSignalFractionInTimeBins.clear();
  TF1 RawResponse("RawResponse", rawResponseFunction, 0, 256, 5);
  RawResponse.SetParameters(1., 0., tau, N, 0.);
  double RawResponseTotalIntegral = (N > 0) ? pow(N, -N) * exp(N) * std::tgamma(N) : 1.;
  for (int j = 0; j < constants::EMCAL_MAXTIMEBINS; j++) {
    double val = RawResponse.Integral(-mTimeBinOffset - 0.5 + j, -mTimeBinOffset + 0.5 + j) / RawResponseTotalIntegral;
    if (val < 1e-10) {
      break;
    }
    mSignalFractionInTimeBins.push_back(val);
  }
}

//_______________________________________________________________________
double Digitizer::rawResponseFunction(double* x, double* par)
{
  double signal = 0.;
  double tau = par[2];
  double n = par[3];
  double ped = par[4];
  double xx = (x[0] - par[1] + tau) / tau;

  if (xx <= 0) {
    signal = ped;
  } else {
    signal = ped + par[0] * std::pow(xx, n) * std::exp(n * (1 - xx));
  }

  return signal;
}

//_______________________________________________________________________
void Digitizer::finish() {}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>& hits, std::vector<Digit>& digits)
{
  digits.clear();
  mDigits.clear();
  mMCTruthContainer.clear();

  if (mSimulateNoiseDigits) {
    addNoiseDigits();
  }

  for (auto hit : hits) {
    try {
      hitToDigits(hit);

      for (auto digit : mTempDigitVector) {
        Int_t id = digit.getTower();

        if (id < 0 || id > mGeometry->GetNCells()) {
          LOG(WARNING) << "tower index out of range: " << id;
          continue;
        }

        MCLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID, false, 1.0);
        LabeledDigit d(digit, label);
        mDigits[id].push_back(d);
      }
    } catch (InvalidPositionException& e) {
      LOG(ERROR) << "Error in creating the digit: " << e.what();
    }
  }

  fillOutputContainer(digits);
}

//_______________________________________________________________________
void Digitizer::hitToDigits(const Hit& hit)
{
  mTempDigitVector.clear();
  Int_t tower = hit.GetDetectorID();
  Double_t energy = hit.GetEnergyLoss();

  if (mSimulateTimeResponse) {
    for (int j = 0; j < mSignalFractionInTimeBins.size(); j++) {
      double val = energy * mSignalFractionInTimeBins.at(j);
      if ((val < mSimParam->getTimeResponseThreshold()) && (j > mTimeBinOffset)) {
        break;
      }
      Digit digit(tower, val, mEventTime + (j - mTimeBinOffset) * constants::EMCAL_TIMESAMPLE);
      mTempDigitVector.push_back(digit);
    }
  } else {
    Digit digit(tower, energy, mEventTime);
    mTempDigitVector.push_back(digit);
  }
}

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
void Digitizer::addNoiseDigits()
{
  for (int id = 0; id <= mGeometry->GetNCells(); id++) {
    double energy = mRandomGenerator->Gaus(0, mSimParam->getPinNoise());
    double time = mRandomGenerator->Rndm() * mSimParam->getTimeNoise();
    Digit digit(id, energy, time);
    MCLabel label(true, 1.0);
    LabeledDigit d(digit, label);
    mDigits[id].push_front(d);
  }
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  std::list<LabeledDigit> l;

  for (auto t : mDigits) {
    std::list<LabeledDigit> tower = t.second;
    tower.sort();

    while (!tower.empty()) {
      LabeledDigit ld1 = tower.front();
      tower.pop_front();

      // loop over all other entries in the container, check if we can add the digits
      std::vector<decltype(tower.begin())> toDelete;
      for (auto ld2 = tower.begin(); ld2 != tower.end(); ++ld2) { // must be iterator in order to know the position in the container for erasing
        if (ld1.canAdd(*ld2)) {
          ld1 += *ld2;
          toDelete.push_back(ld2);
        }
      }
      for (auto del : toDelete) {
        tower.erase(del);
      }

      if (mSmearEnergy) {
        smearEnergy(ld1);
      }

      if (!mRemoveDigitsBelowThreshold || (ld1.getEnergy() >= mSimParam->getDigitThreshold() * (constants::EMCAL_ADCENERGY))) {
        l.push_back(ld1);
      }
    }
  }

  l.sort();

  for (auto d : l) {
    Digit digit = d.getDigit();
    std::vector<MCLabel> labels = d.getLabels();
    digits.push_back(digit);

    Int_t LabelIndex = mMCTruthContainer.getIndexedSize();
    for (auto label : labels) {
      mMCTruthContainer.addElementRandomAccess(LabelIndex, label);
    }
  }
}

//_______________________________________________________________________
void Digitizer::smearEnergy(LabeledDigit& digit)
{
  Double_t energy = digit.getEnergy();
  Double_t fluct = (energy * mSimParam->getMeanPhotonElectron()) / mSimParam->getGainFluctuations();
  energy *= mRandomGenerator->Poisson(fluct) / fluct;
  digit.setEnergy(energy);
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
