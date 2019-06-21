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
#include "EMCALBase/Digit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <climits>
#include <forward_list>
#include <chrono>
#include <TRandom.h>
#include "TF1.h"
#include "FairLogger.h" // for LOG

ClassImp(o2::emcal::Digitizer);

using o2::emcal::Digit;
using o2::emcal::Hit;

using namespace o2::emcal;

//_______________________________________________________________________
void Digitizer::init()
{
  mSimParam = SimParam::GetInstance();
  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  float tau = mSimParam->GetTimeResponseTau();
  float N = mSimParam->GetTimeResponsePower();
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
        Int_t id = digit.GetTower();

        if (id < 0 || id > mGeometry->GetNCells()) {
          LOG(WARNING) << "tower index out of range: " << id << FairLogger::endl;
          continue;
        }

        Int_t LabelIndex = mMCTruthContainer.getIndexedSize();

        Bool_t flag = false;
        for (auto& digit0 : mDigits[id]) {
          if (digit0.canAdd(digit)) {
            digit0 += digit;
            LabelIndex = digit0.GetLabel();
            flag = true;
            break;
          }
        }

        if (!flag) {
          digit.SetLabel(LabelIndex);
          mDigits[id].push_front(digit);
        }

        o2::MCCompLabel label(hit.GetTrackID(), mCurrEvID, mCurrSrcID);
        mMCTruthContainer.addElementRandomAccess(LabelIndex, label);
      }
    } catch (InvalidPositionException& e) {
      LOG(ERROR) << "Error in creating the digit: " << e.what() << FairLogger::endl;
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
      if ((val < mSimParam->GetTimeResponseThreshold()) && (j > mTimeBinOffset)) {
        break;
      }
      Digit digit(tower, val, mEventTime + (j - mTimeBinOffset) * constants::EMCAL_TIMESAMPLE, -9999);
      mTempDigitVector.push_back(digit);
    }
  } else {
    Digit digit(tower, energy, mEventTime, -9999);
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
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")" << FairLogger::endl;
  }
  mEventTime = t;
}

//_______________________________________________________________________
void Digitizer::addNoiseDigits()
{
  for (int id = 0; id <= mGeometry->GetNCells(); id++) {
    double energy = mRandomGenerator->Gaus(0, mSimParam->GetPinNoise());
    double time = mRandomGenerator->Rndm() * mSimParam->GetTimeNoise();
    Digit digit(id, energy, time, -1);
    mDigits[id].push_front(digit);
  }
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  std::forward_list<Digit> l;

  for (auto tower : mDigits) {
    for (auto& digit : tower.second) {
      if (mRemoveDigitsBelowThreshold && (digit.GetAmplitude() < mSimParam->GetDigitThreshold() * (constants::EMCAL_ADCENERGY))) {
        continue;
      }

      if (mSmearEnergy) {
        smearEnergy(digit);
      }

      l.push_front(digit);
    }
  }

  l.sort();

  for (auto digit : l) {
    digits.push_back(digit);
  }
}

//_______________________________________________________________________
void Digitizer::smearEnergy(Digit& digit)
{
  Double_t energy = digit.GetAmplitude();
  Double_t fluct = (energy * mSimParam->GetMeanPhotonElectron()) / mSimParam->GetGainFluctuations();
  energy *= mRandomGenerator->Poisson(fluct) / fluct;
  digit.SetAmplitude(energy);
}

//_______________________________________________________________________
void Digitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if (v > MCCompLabel::maxSourceID()) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label " << MCCompLabel::maxSourceID()
               << FairLogger::endl;
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void Digitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if (v > MCCompLabel::maxEventID()) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label " << MCCompLabel::maxEventID()
               << FairLogger::endl;
  }
  mCurrEvID = v;
}
