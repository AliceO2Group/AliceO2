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

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"

ClassImp(o2::emcal::DigitizerTRU);

using o2::emcal::Digit;
using o2::emcal::Hit;

using namespace o2::emcal;

//_______________________________________________________________________
void DigitizerTRU::init()
{
  setPatches();
  auto& patchTRU = patchesFromAllTRUs[10];
  LOG(info) << "DIG SIMONE init in DigitizerTRU: mPatchIDSeedFastOrIDs[0] = " << std::get<1>(patchTRU.mPatchIDSeedFastOrIDs[0]);
  LOG(info) << "DIG SIMONE init in DigitizerTRU: mPatchIDSeedFastOrIDs[1] = " << std::get<1>(patchTRU.mPatchIDSeedFastOrIDs[1]);
  LOG(info) << "DIG SIMONE init in DigitizerTRU: mPatchIDSeedFastOrIDs[2] = " << std::get<1>(patchTRU.mPatchIDSeedFastOrIDs[2]);




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

  mTriggerMap = new TriggerMappingV2(mGeometry);
  LZERO.setGeometry(mGeometry);
  LZERO.init();
  mAmplitudeInTimeBins.clear();

  // for each phase create a template distribution
  // TF1 RawResponse("RawResponse", rawResponseFunction, 0, 256, 9);
  // RawResponse.SetParameters(3.95714e+03, -4.87952e+03, 2.48989e+03, -6.89067e+02, 1.14413e+02, -1.17744e+01, 7.37825e-01, -2.58461e-02, 3.88652e-04);

  // for (int i = 0; i < 4; i++)
  // {

  //   std::vector<double> sf;
  //   RawResponse.SetParameter(1, 0.25 * i);
  //   for (int j = 0; j < constants::EMCAL_MAXTIMEBINS; j++)
  //   {
  //     sf.push_back(RawResponse.Eval(j - mTimeWindowStart));
  //   }
  //   mAmplitudeInTimeBins.push_back(sf);
  // }

  // Parameters from data (@Martin Poghosyan)
  tau = 61.45; // 61.45 ns, according to the fact that the
               // RawResponse.SetParameter(1, 0.25 * i); where 0.25 are 25 ns
  N = 2.;
  // for each phase create a template distribution
  TF1 RawResponse("RawResponse", rawResponseFunction, 0, 256, 5);
  RawResponse.SetParameters(1., 0., tau, N, 0.);

  // only one phase
  std::vector<double> sf;
  for (int j = 0; j < constants::EMCAL_MAXTIMEBINS; j++) {
    sf.push_back(RawResponse.Eval(j - mTimeWindowStart));
  }
  mAmplitudeInTimeBins.push_back(sf);

  if (mEnableDebugStreaming) {
    mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>("emcaldigitsDebugTRU.root", "RECREATE");
    mDebugStreamPatch = std::make_unique<o2::utils::TreeStreamRedirector>("emcaldigitsDebugPatchTRU.root", "RECREATE");
  }
}

// //_______________________________________________________________________
// double DigitizerTRU::rawResponseFunction(double *x, double *par)
// {
//   double res = par[9 - 1] + par[9] * x[0];
//   for (Int_t j = 9 - 1; j > 0; j--)
//     res = par[j - 1] + x[0] * res;
//   if (x[0] < 2.99)
//     return 0;
//   return res;
// }
//_______________________________________________________________________
double DigitizerTRU::rawResponseFunction(double* x, double* par)
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
void DigitizerTRU::clear()
{
  mDigits.clear();
}
//_______________________________________________________________________
// void DigitizerTRU::process(const std::vector<Digit>& labeledSDigits)
void DigitizerTRU::process(const gsl::span<const Digit> labeledSDigits)
{
  LOG(info) << "DIG SIMONE process in digitizer ";
  // int i = 0;

  auto processedSDigits = makeAnaloguesFastorSums(labeledSDigits);

  for (auto vectorelement : processedSDigits) {
    // for (auto labeleddigit : processedSDigits) {

    int& fastorID = std::get<0>(vectorelement);
    auto& labeleddigit = std::get<1>(vectorelement);

    // LOG(info) << "DIG SIMONE process in digitizer: labeleddigit.getTower ";
    int tower = labeleddigit.getTower();

    // LOG(info) << "DIG SIMONE process in digitizer: before sampleSDigit ";
    // sampleSDigit(labeleddigit.getDigit());
    sampleSDigit(labeleddigit);
    // LOG(info) << "DIG SIMONE process in digitizer: after sampleSDigit ";

    if (mTempDigitVector.size() == 0) {
      continue;
      // LOG(info) << "DIG SIMONE process in digitizer: continue ";
    }

    // LOG(info) << "DIG SIMONE process in digitizer: before addDigits ";
    mDigits.addDigits(fastorID, mTempDigitVector);
    // mDigits.addDigits(tower, mTempDigitVector);
    // LOG(info) << "DIG SIMONE process in digitizer: after addDigits ";
    // ++i;
    // LOG(info) << "DIG SIMONE process in digitizer: after addDigits processed   " << i;
  }
}
//_______________________________________________________________________
// std::vector<Digit> DigitizerTRU::makeAnaloguesFastorSums(const gsl::span<const Digit> sdigits)
std::vector<std::tuple<int, Digit>> DigitizerTRU::makeAnaloguesFastorSums(const gsl::span<const Digit> sdigits)
{
  std::unordered_map<int, Digit> sdigitsFastOR;
  std::vector<int> fastorIndicesFound;
  for (const auto& dig : sdigits) {
    o2::emcal::TriggerMappingV2::IndexCell towerid = dig.getTower();
    int fastorIndex = mTriggerMap->getAbsFastORIndexFromCellIndex(towerid);
    auto found = sdigitsFastOR.find(fastorIndex);
    if (found != sdigitsFastOR.end()) {
      // sum energy
      (found->second) += dig;
    } else {
      // create new digit
      fastorIndicesFound.emplace_back(fastorIndex);
      sdigitsFastOR.emplace(fastorIndex, dig);
    }
  }
  // sort digits for output
  std::sort(fastorIndicesFound.begin(), fastorIndicesFound.end(), std::less<>());
  // std::vector<Digit> outputFastorSDigits;
  // std::for_each(fastorIndicesFound.begin(), fastorIndicesFound.end(), [&outputFastorSDigits, &sdigitsFastOR](int fastorIndex) { outputFastorSDigits.push_back(sdigitsFastOR[fastorIndex]); });
  // return outputFastorSDigits;
  std::vector<std::tuple<int, Digit>> outputFastorSDigits;
  std::for_each(fastorIndicesFound.begin(), fastorIndicesFound.end(), [&outputFastorSDigits, &sdigitsFastOR](int fastorIndex) { outputFastorSDigits.emplace_back(fastorIndex, sdigitsFastOR[fastorIndex]); });
  return outputFastorSDigits;
}

//_______________________________________________________________________
void DigitizerTRU::sampleSDigit(const Digit& sDigit)
{
  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: before mTempDigitVector.clear ";
  mTempDigitVector.clear();
  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: before sDigit.getTower ";
  Int_t tower = sDigit.getTower();
  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: before sDigit.getAmplitude ";
  Double_t energy = sDigit.getAmplitude();

  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: before smearEnergy ";
  if (mSmearEnergy) {
    // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: beforebefore smearEnergy ";
    energy = smearEnergy(energy);
  }

  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: before __DBL_EPSILON__ ";
  if (energy < __DBL_EPSILON__) {
    return;
  }

  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: before TimeResponse ";
  Double_t energies[15];
  if (mSimulateTimeResponse) {
    // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: in TimeResponse ";
    for (int j = 0; j < mAmplitudeInTimeBins.at(0).size(); j++) {
      // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: in TimeResponse mAmplitudeInTimeBins";
      double val = energy * (mAmplitudeInTimeBins.at(0).at(j));
      energies[j] = val;
      // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: in TimeResponse digitTime";
      double digitTime = (mEventTimeOffset + mDelay - mTimeWindowStart) * constants::EMCAL_TIMESAMPLE;
      Digit digit(tower, val, digitTime);
      // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: in TimeResponse push_back";
      mTempDigitVector.push_back(digit);
    }
  } else {
    Digit digit(tower, energy, (mDelay - mTimeWindowStart) * constants::EMCAL_TIMESAMPLE);
    mTempDigitVector.push_back(digit);
  }

  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: before Debug ";
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
  // LOG(info) << "DIG SIMONE sampleSDigit in digitizer: after Debug ";
}

//_______________________________________________________________________
double DigitizerTRU::smearEnergy(double energy)
{
  // LOG(info) << "DIG SIMONE smearEnergy in digitizer: after Debug ";
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


  if (mEnableDebugStreaming) {
    LOG(info) << "DIG SIMONE setEventTime in digitizer: before  mEnableDebugStreaming";
    auto TriggerInputs        = LZERO.getTriggerInputs();
    auto TriggerInputsPatches = LZERO.getTriggerInputsPatches();
    int nIter = TriggerInputs.size(); 
    // for (int i = 0; i < nIter; i++)
    // {
    //   auto allfastOrs = TriggerInputs[i].mLastTimesumAllFastOrs;
    //   auto nFastOrs = allfastOrs.size();
    //   double FastOrAmp[nFastOrs];
    //   int WhichTRU[nFastOrs];
    //   int WhichFastOr[nFastOrs];
    //   (*mDebugStream).GetFile()->cd();
    //   for (int j = 0; j < nFastOrs; j++){
    //     WhichTRU[j]    = std::get<0>(allfastOrs[i]);
    //     WhichFastOr[j] = std::get<1>(allfastOrs[i]);
    //     FastOrAmp[j]   = std::get<2>(allfastOrs[i]);
    //     (*mDebugStream) << "L0Timesums" 
    //     << "WhichTRU=" << WhichTRU[j]
    //     << "WhichFastOr=" << WhichFastOr[j]
    //     << "FastOrAmp=" << FastOrAmp[j]
    //     << "\n";
    //   }
    //   LOG(info) << "DIG SIMONE setEventTime in digitizer: fill TREE";
      
    // }
    if(nIter != 0){
    LOG(info) << "DIG SIMONE setEventTime in digitizer: size of TriggerInputs = " << nIter;
    for (auto& trigger : TriggerInputs)
    {
      // LOG(info) << "DIG SIMONE setEventTime in digitizer: before  loop TriggerInputs";
      // LOG(info) << "DIG SIMONE setEventTime in digitizer: size of TriggerInputs = " << TriggerInputs.size();
      LOG(info) << "DIG SIMONE setEventTime in digitizer: size of trigger.mLastTimesumAllFastOrs = " << trigger.mLastTimesumAllFastOrs.size();
      for(auto& fastor : trigger.mLastTimesumAllFastOrs){
        LOG(info) << "DIG SIMONE setEventTime in digitizer: inside loop";
        auto WhichTRU    = std::get<0>(fastor);
        auto WhichFastOr = std::get<1>(fastor);
        auto FastOrAmp   = std::get<2>(fastor);
        LOG(info) << "DIG SIMONE setEventTime in digitizer: before filling";
        (*mDebugStream).GetFile()->cd();
        (*mDebugStream) << "L0Timesums" 
          << "WhichTRU=" << WhichTRU
          << "WhichFastOr=" << WhichFastOr
          << "FastOrAmp=" << FastOrAmp
          << "\n";
        LOG(info) << "DIG SIMONE setEventTime in digitizer: fill TREE";
      }
    }
    for (auto& trigger : TriggerInputsPatches)
    {
      // LOG(info) << "DIG SIMONE setEventTime in digitizer: before  lastTimeSum";
      auto lastTimeSum  = trigger.mLastTimesumAllPatches.end()-1;
      // LOG(info) << "DIG SIMONE setEventTime in digitizer: before  loop";
      LOG(info) << "DIG SIMONE setEventTime in digitizer: size of trigger.mLastTimesumAllPatches = " << trigger.mLastTimesumAllPatches.size();
      for(auto& patches : trigger.mLastTimesumAllPatches){
        LOG(info) << "DIG SIMONE setEventTime in digitizer: inside loop";
        auto WhichTRU     = std::get<0>(patches);
        auto WhichPatch   = std::get<1>(patches);
        auto PatchTimesum = std::get<2>(patches);
        LOG(info) << "DIG SIMONE setEventTime in digitizer: before  isFired";
        auto isFired      = std::get<3>(patches);
        // (*mDebugStream).GetFile()->cd();
        // (*mDebugStream) << "L0TimesumsPatch" 
        LOG(info) << "DIG SIMONE setEventTime in digitizer: before  GetFile()";
        (*mDebugStreamPatch).GetFile()->cd();
        (*mDebugStreamPatch) << "L0TimesumsPatch" 
          << "WhichTRU=" << WhichTRU
          << "WhichPatch=" << WhichPatch
          << "PatchTimesum=" << PatchTimesum
          << "isFired=" << isFired
          << "\n";
        LOG(info) << "DIG SIMONE setEventTime in digitizer: fill TREE per patch";
      }
    }
    }
  }
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
  Patches FullAside(2, 0, 0);
  Patches FullCside(2, 1, 0);
  Patches ThirdAside(2, 0, 1);
  Patches ThirdCside(2, 1, 1);
  FullAside.init();
  FullCside.init();
  ThirdAside.init();
  ThirdCside.init();
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: FullAside.mPatchIDSeedFastOrIDs[0] = " << std::get<1>(FullAside.mPatchIDSeedFastOrIDs[0]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: FullAside.mPatchIDSeedFastOrIDs[1] = " << std::get<1>(FullAside.mPatchIDSeedFastOrIDs[1]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: FullAside.mPatchIDSeedFastOrIDs[2] = " << std::get<1>(FullAside.mPatchIDSeedFastOrIDs[2]);

  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: FullCside.mPatchIDSeedFastOrIDs[0] = " << std::get<1>(FullCside.mPatchIDSeedFastOrIDs[0]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: FullCside.mPatchIDSeedFastOrIDs[1] = " << std::get<1>(FullCside.mPatchIDSeedFastOrIDs[1]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: FullCside.mPatchIDSeedFastOrIDs[2] = " << std::get<1>(FullCside.mPatchIDSeedFastOrIDs[2]);

  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: ThirdAside.mPatchIDSeedFastOrIDs[0] = " << std::get<1>(ThirdAside.mPatchIDSeedFastOrIDs[0]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: ThirdAside.mPatchIDSeedFastOrIDs[1] = " << std::get<1>(ThirdAside.mPatchIDSeedFastOrIDs[1]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: ThirdAside.mPatchIDSeedFastOrIDs[2] = " << std::get<1>(ThirdAside.mPatchIDSeedFastOrIDs[2]);

  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: ThirdCside.mPatchIDSeedFastOrIDs[0] = " << std::get<1>(ThirdCside.mPatchIDSeedFastOrIDs[0]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: ThirdCside.mPatchIDSeedFastOrIDs[1] = " << std::get<1>(ThirdCside.mPatchIDSeedFastOrIDs[1]);
  LOG(info) << "DIG SIMONE setPatches in DigitizerTRU: ThirdCside.mPatchIDSeedFastOrIDs[2] = " << std::get<1>(ThirdCside.mPatchIDSeedFastOrIDs[2]);

  // EMCAL
  for (int i = 0; i < 5; i++) {
    for (int i = 0; i < 3; i++)
      patchesFromAllTRUs.push_back(FullAside);
    for (int i = 0; i < 3; i++)
      patchesFromAllTRUs.push_back(FullCside);
  }
  patchesFromAllTRUs.push_back(ThirdAside);
  patchesFromAllTRUs.push_back(ThirdCside);

  // DCAL
  for (int i = 0; i < 3; i++) {
    for (int i = 0; i < 2; i++)
      patchesFromAllTRUs.push_back(FullAside);
    for (int i = 0; i < 2; i++)
      patchesFromAllTRUs.push_back(FullCside);
  }
  patchesFromAllTRUs.push_back(ThirdAside);
  patchesFromAllTRUs.push_back(ThirdCside);
}
//______________________________________________________________________
void DigitizerTRU::finish() { 
  mDigits.finish(); 
  if ( isDebugMode() == true ){
    endDebugStream();
  }
}