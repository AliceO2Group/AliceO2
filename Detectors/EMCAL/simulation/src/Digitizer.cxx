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

#include "EMCALSimulation/Digitizer.h"
#include "EMCALSimulation/SimParam.h"
#include "EMCALSimulation/DigitsWriteoutBuffer.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Hit.h"
#include "MathUtils/Cartesian.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "EMCALSimulation/DigitsWriteoutBuffer.h"
#include <climits>
#include <forward_list>
#include <chrono>
#include <TRandom.h>
#include <TF1.h>
#include "FairLogger.h" // for LOG
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonUtils/TreeStreamRedirector.h"

ClassImp(o2::emcal::Digitizer);

using o2::emcal::Digit;
using o2::emcal::Hit;

using namespace o2::emcal;

//_______________________________________________________________________
void Digitizer::init()
{
  mSimParam = &(o2::emcal::SimParam::Instance());
  mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  float tau = mSimParam->getTimeResponseTau();
  float N = mSimParam->getTimeResponsePower();
  float delay = std::fmod(mSimParam->getSignalDelay() / constants::EMCAL_TIMESAMPLE, 1);
  mDelay = ((int)(std::floor(mSimParam->getSignalDelay() / constants::EMCAL_TIMESAMPLE)));
  mTimeWindowStart = ((unsigned int)(std::floor(mSimParam->getTimeBinOffset() / constants::EMCAL_TIMESAMPLE)));

  mSmearEnergy = mSimParam->doSmearEnergy();
  mSimulateTimeResponse = mSimParam->doSimulateTimeResponse();

  mDigits.init();
  if ((mDelay - mTimeWindowStart) != 0) {
    mDigits.setBufferSize(mDigits.getBufferSize() + (mDelay - mTimeWindowStart));
    mDigits.reserve();
  }

  mAmplitudeInTimeBins.clear();

  // for each phase create a template distribution
  TF1 RawResponse("RawResponse", rawResponseFunction, 0, 256, 5);
  RawResponse.SetParameters(1., 0., tau, N, 0.);

  for (int i = 0; i < 4; i++) {

    std::vector<double> sf;
    RawResponse.SetParameter(1, 0.25 * i);
    for (int j = 0; j < constants::EMCAL_MAXTIMEBINS; j++) {
      sf.push_back(RawResponse.Eval(j - mTimeWindowStart));
    }
    mAmplitudeInTimeBins.push_back(sf);
  }

  if (mEnableDebugStreaming) {
    mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>("emcaldigitsDebug.root", "RECREATE");
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

  // par[0] amp, par[1] peak time

  if (xx <= 0) {
    signal = ped;
  } else {
    signal = ped + par[0] * std::pow(xx, n) * std::exp(n * (1 - xx));
  }

  return signal;
}

//_______________________________________________________________________
void Digitizer::clear()
{
  mDigits.clear();
}

//_______________________________________________________________________
void Digitizer::process(const std::vector<LabeledDigit>& labeledSDigits)
{

  for (auto labeleddigit : labeledSDigits) {

    int tower = labeleddigit.getTower();

    sampleSDigit(labeleddigit.getDigit());

    if (mTempDigitVector.size() == 0) {
      continue;
    }

    std::vector<LabeledDigit> listofLabeledDigit;

    for (auto& digit : mTempDigitVector) {
      Int_t id = digit.getTower();

      auto labels = labeleddigit.getLabels();
      LabeledDigit d(digit, labels[0]);
      int iLabel(0);
      for (auto& label : labels) {
        if (digit.getAmplitude() < __DBL_EPSILON__) {
          label.setAmplitudeFraction(0);
        }
        if (iLabel == 0) {
          continue;
        }
        d.addLabel(label);
        iLabel++;
      }
      listofLabeledDigit.push_back(d);
    }
    mDigits.addDigits(tower, listofLabeledDigit);
  }
}

//_______________________________________________________________________
void Digitizer::sampleSDigit(const Digit& sDigit)
{
  mTempDigitVector.clear();
  Int_t tower = sDigit.getTower();
  Double_t energy = sDigit.getAmplitude();

  if (mSmearEnergy) {
    energy = smearEnergy(energy);
  }

  if (energy < __DBL_EPSILON__) {
    return;
  }

  Double_t energies[15];
  if (mSimulateTimeResponse) {
    for (int j = 0; j < mAmplitudeInTimeBins.at(mPhase).size(); j++) {

      double val = energy * (mAmplitudeInTimeBins.at(mPhase).at(j));
      energies[j] = val;
      double digitTime = (mEventTimeOffset + mDelay - mTimeWindowStart) * constants::EMCAL_TIMESAMPLE;
      Digit digit(tower, val, digitTime);
      mTempDigitVector.push_back(digit);
    }
  } else {
    Digit digit(tower, energy, (mDelay - mTimeWindowStart) * constants::EMCAL_TIMESAMPLE);
    mTempDigitVector.push_back(digit);
  }

  if (mEnableDebugStreaming) {
    double timeStamp = sDigit.getTimeStamp();
    (*mDebugStream).GetFile()->cd();
    (*mDebugStream) << "DigitsTimeSamples"
                    << "Tower=" << tower
                    << "Time=" << timeStamp
                    << "DigitEnergy=" << energy
                    << "Sample0=" << energies[0]
                    << "Sample1=" << energies[1]
                    << "Sample2=" << energies[2]
                    << "Sample3=" << energies[3]
                    << "Sample4=" << energies[4]
                    << "Sample5=" << energies[5]
                    << "Sample6=" << energies[6]
                    << "Sample7=" << energies[7]
                    << "Sample8=" << energies[8]
                    << "Sample9=" << energies[9]
                    << "Sample10=" << energies[10]
                    << "Sample11=" << energies[11]
                    << "Sample12=" << energies[12]
                    << "Sample13=" << energies[13]
                    << "Sample14=" << energies[14]
                    << "\n";
  }
}

//_______________________________________________________________________
double Digitizer::smearEnergy(double energy)
{
  Double_t fluct = (energy * mSimParam->getMeanPhotonElectron()) / mSimParam->getGainFluctuations();
  energy *= mRandomGenerator->Poisson(fluct) / fluct;
  return energy;
}

//_______________________________________________________________________
void Digitizer::setEventTime(o2::InteractionTimeRecord record)
{

  mDigits.forwardMarker(record);

  mPhase = mDigits.getPhase();

  mEventTimeOffset = 0;

  if (mPhase == 4) {
    mPhase = 0;
    mEventTimeOffset++;
  }
}