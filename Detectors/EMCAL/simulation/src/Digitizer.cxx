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
  mSimParam = &(o2::emcal::SimParam::Instance());
  mLiveTime = mSimParam->getLiveTime();
  mBusyTime = mSimParam->getBusyTime();
  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  float tau = mSimParam->getTimeResponseTau();
  float N = mSimParam->getTimeResponsePower();
  float delay = std::fmod(mSimParam->getSignalDelay() / constants::EMCAL_TIMESAMPLE, 1);
  mDelay = ((int)(std::floor(mSimParam->getSignalDelay() / constants::EMCAL_TIMESAMPLE)));

  mTimeBinOffset.clear();
  mSignalFractionInTimeBins.clear();

  TF1 RawResponse("RawResponse", rawResponseFunction, 0, 256, 5);
  RawResponse.SetParameters(1., 0., tau, N, 0.);

  for (int i = 0; i < 4; i++) {
    int offset = ((int)(std::ceil(tau - delay - 0.25 * i)));
    mTimeBinOffset.push_back(offset);

    double RawResponseTotalIntegral = (N > 0) ? pow(N / tau, -N) * exp(N) * std::tgamma(N) * std::pow(1 / tau, N - 1) : 1.;
    std::vector<double> sf;
    RawResponse.SetParameter(1, 0.25 * i + delay);

    for (int j = 0; j < constants::EMCAL_MAXTIMEBINS; j++) {
      double val = RawResponse.Integral(j - offset, j + 1 - offset) / RawResponseTotalIntegral;
      sf.push_back(val);
    }

    mSignalFractionInTimeBins.push_back(sf);
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
void Digitizer::initCycle()
{
  if (mSimulateNoiseDigits) {
    addNoiseDigits();
  }

  mEmpty = false;
}

//_______________________________________________________________________
void Digitizer::clear()
{
  mTriggerTime = -1e20;
  mDigits.clear();
  mEmpty = true;
}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>& hits)
{
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

  mEmpty = false;
}

//_______________________________________________________________________
void Digitizer::hitToDigits(const Hit& hit)
{
  mTempDigitVector.clear();
  Int_t tower = hit.GetDetectorID();
  Double_t energy = hit.GetEnergyLoss();

  if (mSimulateTimeResponse) {
    for (int j = 0; j < mSignalFractionInTimeBins.at(mPhase).size(); j++) {
      double val = energy * (mSignalFractionInTimeBins.at(mPhase).at(j));
      if ((val < mSimParam->getTimeResponseThreshold()) && (j > mTimeBinOffset.at(mPhase))) {
        break;
      }
      Digit digit(tower, val, mEventTime + (j - mTimeBinOffset.at(mPhase) + mDelay + 0.5) * constants::EMCAL_TIMESAMPLE);
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

  if (t - mTriggerTime >= mLiveTime + mBusyTime) {
    mTriggerTime = t;
  }

  mEventTime = t - mTriggerTime;

  mPhase = ((int)((std::fmod(t, 100) + 12.5) / 25));
  if (mPhase == 4) {
    mPhase = 0;
  }
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
void Digitizer::fillOutputContainer(std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>& labelsout)
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

    Int_t LabelIndex = labelsout.getIndexedSize();
    for (auto label : labels) {
      labelsout.addElementRandomAccess(LabelIndex, label);
    }
  }

  mDigits.clear();
  mEmpty = true;
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
