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

#include "EMCALSimulation/DigitizerTRU.h"
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
#include <fairlogger/Logger.h> // for LOG
#include "CommonDataFormat/InteractionRecord.h"
#include "CommonUtils/TreeStreamRedirector.h"

ClassImp(o2::emcal::DigitizerTRU);

using o2::emcal::Digit;
using o2::emcal::Hit;

using namespace o2::emcal;

//_______________________________________________________________________
void DigitizerTRU::init()
{
  setPatches();

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
  mDigits.reserve(15);
  // if ((mDelay - mTimeWindowStart) != 0)
  // {
  //   mDigits.setBufferSize(mDigits.getBufferSize() + (mDelay - mTimeWindowStart));
  //   mDigits.reserve();
  // }

  mAmplitudeInTimeBins.clear();

  // for each phase create a template distribution
  TF1 RawResponse("RawResponse", rawResponseFunction, 0, 256, 9);
  RawResponse.SetParameters(3.95714e+03, -4.87952e+03, 2.48989e+03, -6.89067e+02, 1.14413e+02, -1.17744e+01, 7.37825e-01, -2.58461e-02, 3.88652e-04);

  for (int i = 0; i < 4; i++)
  {

    std::vector<double> sf;
    RawResponse.SetParameter(1, 0.25 * i);
    for (int j = 0; j < constants::EMCAL_MAXTIMEBINS; j++)
    {
      sf.push_back(RawResponse.Eval(j - mTimeWindowStart));
    }
    mAmplitudeInTimeBins.push_back(sf);
  }

  if (mEnableDebugStreaming)
  {
    mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>("emcaldigitsDebug.root", "RECREATE");
  }
}

//_______________________________________________________________________
double DigitizerTRU::rawResponseFunction(double *x, double *par)
{
  double res = par[9 - 1] + par[9] * x[0];
  for (Int_t j = 9 - 1; j > 0; j--)
    res = par[j - 1] + x[0] * res;
  if (x[0] < 2.99)
    return 0;
  return res;
}
//_______________________________________________________________________
void DigitizerTRU::clear()
{
  mDigits.clear();
}
//_______________________________________________________________________
// void DigitizerTRU::process(const std::vector<LabeledDigit> &labeledSDigits)
void DigitizerTRU::process(const std::vector<Digit> &labeledSDigits)
{

  for (auto labeleddigit : labeledSDigits)
  {

    int tower = labeleddigit.getTower();

    // sampleSDigit(labeleddigit.getDigit());
    sampleSDigit(labeleddigit);

    if (mTempDigitVector.size() == 0)
    {
      continue;
    }

    mDigits.addDigits(tower, mTempDigitVector);
  }
}
//_______________________________________________________________________
void DigitizerTRU::sampleSDigit(const Digit &sDigit)
{
  mTempDigitVector.clear();
  Int_t tower = sDigit.getTower();
  Double_t energy = sDigit.getAmplitude();

  if (mSmearEnergy)
  {
    energy = smearEnergy(energy);
  }

  if (energy < __DBL_EPSILON__)
  {
    return;
  }

  Double_t energies[15];
  if (mSimulateTimeResponse)
  {
    for (int j = 0; j < mAmplitudeInTimeBins.at(mPhase).size(); j++)
    {

      double val = energy * (mAmplitudeInTimeBins.at(mPhase).at(j));
      energies[j] = val;
      double digitTime = (mEventTimeOffset + mDelay - mTimeWindowStart) * constants::EMCAL_TIMESAMPLE;
      Digit digit(tower, val, digitTime);
      mTempDigitVector.push_back(digit);
    }
  }
  else
  {
    Digit digit(tower, energy, (mDelay - mTimeWindowStart) * constants::EMCAL_TIMESAMPLE);
    mTempDigitVector.push_back(digit);
  }

  if (mEnableDebugStreaming)
  {
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
double DigitizerTRU::smearEnergy(double energy)
{
  Double_t fluct = (energy * mSimParam->getMeanPhotonElectron()) / mSimParam->getGainFluctuations();
  energy *= mRandomGenerator->Poisson(fluct) / fluct;
  return energy;
}

//_______________________________________________________________________
void DigitizerTRU::setEventTime(o2::InteractionTimeRecord record)
{
  // For the digitisation logic, at this time you would do:
  // mDigits.forwardMarker(record);
  // In the case of the trigger simulation, what is needed is to
  // fill the corresponding LZEROElectronics object and
  // launch the peak finder.
  // If a trigger is found the logic is set to be live.
  mDigits.fillOutputContainer(false, record, patchesFromAllTRUs, LZERO);


  // mPhase = mSimParam->doSimulateL1Phase() ? mDigits.getPhase() : 0;

  // mEventTimeOffset = 0;

  // if (mPhase == 4)
  // {
  //   mPhase = 0;
  //   mEventTimeOffset++;
  // }
}
//_______________________________________________________________________
void DigitizerTRU::setPatches()
{
  // Using Run 2 geometry, found in:
  // https://www.dropbox.com/s/pussj0olcctroim/PHOS+DCAL_STU.pdf?dl=0
  // TRUs[0+6n, 1+6n, 2+6n] are on the A-side
  // TRUs[3+6n, 4+6n, 5+6n] are on the C-side

  patchesFromAllTRUs.clear();
  // patchesFromAllTRUs.resize();
  Patches FullAside(2,0,0);
  Patches FullCside(2,1,0);
  Patches ThirdAside(2,0,1);
  Patches ThirdCside(2,1,1);
  FullAside.init();
  FullCside.init();
  ThirdAside.init();
  ThirdCside.init();

  // EMCAL
  for(int i = 0; i < 5; i++){
    for(int i = 0; i < 3; i++) patchesFromAllTRUs.push_back(FullAside);
    for(int i = 0; i < 3; i++) patchesFromAllTRUs.push_back(FullCside);
  }
  patchesFromAllTRUs.push_back(ThirdAside);
  patchesFromAllTRUs.push_back(ThirdCside);

  // DCAL
  for(int i = 0; i < 3; i++){
    for(int i = 0; i < 2; i++) patchesFromAllTRUs.push_back(FullAside);
    for(int i = 0; i < 2; i++) patchesFromAllTRUs.push_back(FullCside);
  }
  patchesFromAllTRUs.push_back(ThirdAside);
  patchesFromAllTRUs.push_back(ThirdCside);



}