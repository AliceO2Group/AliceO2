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
  auto mSimParam = &(o2::emcal::SimParam::Instance());
  // mRandomGenerator = new TRandom3(std::chrono::high_resolution_clock::now().time_since_epoch().count());

  float tau = mSimParam->getTimeResponseTauTRU();
  float N = mSimParam->getTimeResponsePowerTRU();

  mSmearEnergy = mSimParam->doSmearEnergy();
  mSimulateTimeResponse = mSimParam->doSimulateTimeResponse();

  mDigits.init();
  mDigits.reserve(15);

  LZERO.setGeometry(mGeometry);
  LZERO.init();

  // Parameters from data (@Martin Poghosyan)
  // tau = 61.45 / 25.; // 61.45 ns, according to the fact that the
  // N = 2.;

  if (mSimulateTimeResponse) {
    // for each phase create a template distribution
    TF1 RawResponse("RawResponse", rawResponseFunction, 0, 256, 5);
    RawResponse.SetParameters(1., 0., tau, N, 0.);
    RawResponse.SetParameter(1, 425. / o2::emcal::constants::EMCAL_TIMESAMPLE);

    // parameter 1: Handling phase + delay
    // phase: 25 ns * phase index (-4)
    // delay: Average signal delay
    RawResponse.SetParameter(1, mSimParam->getSignalDelay() / constants::EMCAL_TIMESAMPLE);
    for (int sample = 0; sample < constants::EMCAL_MAXTIMEBINS; sample++) {
      mAmplitudeInTimeBins[sample] = RawResponse.Eval(sample);
      LOG(info) << "DIG TRU init in DigitizerTRU: amplitudes[" << sample << "] = " << mAmplitudeInTimeBins[sample];
    }

    // Rescale the TRU time response function
    // so that the sum of the samples at [5], [6], [7] and [8]
    // is equal to 1
    auto maxElement = std::max_element(mAmplitudeInTimeBins.begin(), mAmplitudeInTimeBins.end());
    double rescalingFactor = *(maxElement - 1) + *maxElement + *(maxElement + 1) + *(maxElement + 2);
    double normalisationTRU = mSimParam->getTimeResponseNormalisationTRU();
    rescalingFactor /= normalisationTRU;
    // rescalingFactor /= (5. / 3.93);   // the slope seen in the single fastOr correlation
    // rescalingFactor /= (45. / 39.25); // the slope seen in the single fastOr correlation
    for (int sample = 0; sample < constants::EMCAL_MAXTIMEBINS; sample++) {
      mAmplitudeInTimeBins[sample] /= rescalingFactor;
      LOG(info) << "DIG TRU init in DigitizerTRU after RESCALING: amplitudes[" << sample << "] = " << mAmplitudeInTimeBins[sample];
    }
  } else {
  }

  if (mEnableDebugStreaming) {
    mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>("emcaldigitsDebugTRU.root", "RECREATE");
    // mDebugStreamPatch = std::make_unique<o2::utils::TreeStreamRedirector>("emcaldigitsDebugPatchTRU.root", "RECREATE");
  }
}
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
void DigitizerTRU::process(const gsl::span<const Digit> summableDigits)
{

  auto processedSDigits = makeAnaloguesFastorSums(summableDigits);

  for (auto vectorelement : processedSDigits) {

    int& fastorID = std::get<0>(vectorelement);
    auto& digit = std::get<1>(vectorelement);

    int tower = digit.getTower();

    sampleSDigit(digit);

    if (mTempDigitVector.size() == 0) {
      continue;
    }

    mDigits.addDigits(fastorID, mTempDigitVector);
  }
}
//_______________________________________________________________________
std::vector<std::tuple<int, Digit>> DigitizerTRU::makeAnaloguesFastorSums(const gsl::span<const Digit> sdigits)
{
  TriggerMappingV2 mTriggerMap(mGeometry);
  std::unordered_map<int, Digit> sdigitsFastOR;
  std::vector<int> fastorIndicesFound;
  for (const auto& dig : sdigits) {
    o2::emcal::TriggerMappingV2::IndexCell towerid = dig.getTower();
    int fastorIndex = mTriggerMap.getAbsFastORIndexFromCellIndex(towerid);
    auto whichTRU = std::get<0>(mTriggerMap.getTRUFromAbsFastORIndex(fastorIndex));
    auto whichFastOrTRU = std::get<1>(mTriggerMap.getTRUFromAbsFastORIndex(fastorIndex));
    auto found = sdigitsFastOR.find(fastorIndex);
    if (found != sdigitsFastOR.end()) {
      // sum energy
      Digit digitToSum((found->second).getTower(), dig.getAmplitude(), (found->second).getTimeStamp());
      (found->second) += digitToSum;
    } else {
      // create new digit
      fastorIndicesFound.emplace_back(fastorIndex);
      sdigitsFastOR.emplace(fastorIndex, dig);
    }
  }
  // sort digits for output
  std::sort(fastorIndicesFound.begin(), fastorIndicesFound.end(), std::less<>());

  for (auto& elem : sdigitsFastOR) {
    auto dig = elem.second;
    int fastorIndex = elem.first;
    auto whichTRU = std::get<0>(mTriggerMap.getTRUFromAbsFastORIndex(fastorIndex));
    auto whichFastOrTRU = std::get<1>(mTriggerMap.getTRUFromAbsFastORIndex(fastorIndex));
  }

  // Setting them to be TRU digits
  for (auto& elem : sdigitsFastOR) {
    (elem.second).setTRU();
  }

  for (auto& elem : sdigitsFastOR) {
    auto dig = elem.second;
    int fastorIndex = elem.first;
    auto whichTRU = std::get<0>(mTriggerMap.getTRUFromAbsFastORIndex(fastorIndex));
    auto whichFastOrTRU = std::get<1>(mTriggerMap.getTRUFromAbsFastORIndex(fastorIndex));
  }

  std::vector<std::tuple<int, Digit>> outputFastorSDigits;
  std::for_each(fastorIndicesFound.begin(), fastorIndicesFound.end(), [&outputFastorSDigits, &sdigitsFastOR](int fastorIndex) { outputFastorSDigits.emplace_back(fastorIndex, sdigitsFastOR[fastorIndex]); });
  return outputFastorSDigits;
}

//_______________________________________________________________________
void DigitizerTRU::sampleSDigit(const Digit& sDigit)
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
    for (int sample = 0; sample < mAmplitudeInTimeBins.size(); sample++) {

      double val = energy * (mAmplitudeInTimeBins[sample]);
      energies[sample] = val;
      double digitTime = mEventTimeOffset * constants::EMCAL_TIMESAMPLE;
      Digit digit(tower, val, digitTime);
      digit.setTRU();
      mTempDigitVector.push_back(digit);
    }

  } else {
    Digit digit(tower, energy, smearTime(sDigit.getTimeStamp(), energy));
    digit.setTRU();
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
double DigitizerTRU::smearEnergy(double energy)
{
  auto mSimParam = &(o2::emcal::SimParam::Instance());
  Double_t fluct = (energy * mSimParam->getMeanPhotonElectron()) / mSimParam->getGainFluctuations();
  TRandom3 mRandomGenerator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  energy *= mRandomGenerator.Poisson(fluct) / fluct;
  return energy;
}
//_______________________________________________________________________
double DigitizerTRU::smearTime(double time, double energy)
{
  auto mSimParam = &(o2::emcal::SimParam::Instance());
  TRandom3 mRandomGenerator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  return mRandomGenerator.Gaus(time + mSimParam->getSignalDelay(), mSimParam->getTimeResolution(energy));
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
  mEventTimeOffset = 0;

  if (mEnableDebugStreaming) {
    auto TriggerInputsAll = LZERO.getTriggerInputs();
    auto TriggerInputsPatchesAll = LZERO.getTriggerInputsPatches();

    std::vector<o2::emcal::EMCALTriggerInputs> TriggerInputs;
    if (TriggerInputsAll.size() != mPreviousTriggerSize) {
      mWasTriggerFound = true;
      mPreviousTriggerSize = TriggerInputsAll.size();
    } else {
      mWasTriggerFound = false;
    }
    if (TriggerInputsAll.size() > 0 && mWasTriggerFound == true) {
      TriggerInputs.push_back(TriggerInputsAll.back());
    }
    std::vector<o2::emcal::EMCALTriggerInputsPatch> TriggerInputsPatches;
    if (TriggerInputsPatchesAll.size() > 0 && mWasTriggerFound == true) {
      TriggerInputsPatches.push_back(TriggerInputsPatchesAll.back());
    }
    int nIter = TriggerInputs.size();

    if (nIter != 0) {
      for (auto& trigger : TriggerInputs) {
        auto InteractionRecordData = trigger.mInterRecord;
        auto bc = InteractionRecordData.bc;
        auto orbit = InteractionRecordData.orbit;
        for (auto& fastor : trigger.mLastTimesumAllFastOrs) {
          auto WhichTRU = std::get<0>(fastor);
          auto WhichFastOr = std::get<1>(fastor);
          auto FastOrAmp = std::get<2>(fastor);
          (*mDebugStream).GetFile()->cd();
          (*mDebugStream) << "L0Timesums"
                          << "bc=" << bc
                          << "orbit=" << orbit
                          << "WhichTRU=" << WhichTRU
                          << "WhichFastOr=" << WhichFastOr
                          << "FastOrAmp=" << FastOrAmp
                          << "\n";
        }
      }
      // for (auto& trigger : TriggerInputsPatches) {
      //   auto lastTimeSum = trigger.mLastTimesumAllPatches.end() - 1;
      //   for (auto& patches : trigger.mLastTimesumAllPatches) {
      //     auto WhichTRU = std::get<0>(patches);
      //     auto WhichPatch = std::get<1>(patches);
      //     auto PatchTimesum = std::get<2>(patches);
      //     auto isFired = std::get<3>(patches);
      //     (*mDebugStreamPatch).GetFile()->cd();
      //     (*mDebugStreamPatch) << "L0TimesumsPatch"
      //                          << "WhichTRU=" << WhichTRU
      //                          << "WhichPatch=" << WhichPatch
      //                          << "PatchTimesum=" << PatchTimesum
      //                          << "isFired=" << isFired
      //                          << "\n";
      //   }
      // }
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
  TRUElectronics FullAside(2, 0, 0);
  TRUElectronics FullCside(2, 1, 0);
  TRUElectronics ThirdAside(2, 0, 1);
  TRUElectronics ThirdCside(2, 1, 1);
  FullAside.init();
  FullCside.init();
  ThirdAside.init();
  ThirdCside.init();

  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 0,1,2    EMCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 3,4,5    EMCAL C-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 6,7,8    EMCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 9,10,11  EMCAL C-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 12,13,14 EMCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 15,16,17 EMCAL C-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 18,19,20 EMCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 21,22,23 EMCAL C-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 24,25,26 EMCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 27,28,29 EMCAL C-side, Full
  }
  patchesFromAllTRUs.push_back(ThirdAside); // TRU ID 30 EMCAL A-side, Third
  patchesFromAllTRUs.push_back(ThirdCside); // TRU ID 31 EMCAL C-side, Third
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 32,33,34 DCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 35,36,37 DCAL C-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 38,39,40 DCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 41,42,43 DCAL C-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullAside); // TRU ID 44,45,46 DCAL A-side, Full
  }
  for (int j = 0; j < 3; j++) {
    patchesFromAllTRUs.push_back(FullCside); // TRU ID 47,48,49 DCAL C-side, Full
  }
  patchesFromAllTRUs.push_back(ThirdAside); // TRU ID 50 DCAL A-side, Third
  patchesFromAllTRUs.push_back(ThirdCside); // TRU ID 51 DCAL C-side, Third

  while (patchesFromAllTRUs[30].mPatchIDSeedFastOrIDs.size() > 69) {
    patchesFromAllTRUs[30].mPatchIDSeedFastOrIDs.pop_back();
    patchesFromAllTRUs[30].mIndexMapPatch.pop_back();
  }
  while (patchesFromAllTRUs[31].mPatchIDSeedFastOrIDs.size() > 69) {
    patchesFromAllTRUs[31].mPatchIDSeedFastOrIDs.pop_back();
    patchesFromAllTRUs[31].mIndexMapPatch.pop_back();
  }
  while (patchesFromAllTRUs[50].mPatchIDSeedFastOrIDs.size() > 69) {
    patchesFromAllTRUs[50].mPatchIDSeedFastOrIDs.pop_back();
    patchesFromAllTRUs[50].mIndexMapPatch.pop_back();
  }
  while (patchesFromAllTRUs[51].mPatchIDSeedFastOrIDs.size() > 69) {
    patchesFromAllTRUs[51].mPatchIDSeedFastOrIDs.pop_back();
    patchesFromAllTRUs[51].mIndexMapPatch.pop_back();
  }
}
//______________________________________________________________________
void DigitizerTRU::finish()
{
  mDigits.finish();
  if (isDebugMode() == true) {
    endDebugStream();
  }
}