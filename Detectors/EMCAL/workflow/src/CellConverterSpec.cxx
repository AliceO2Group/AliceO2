// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <gsl/span>
#include "FairLogger.h"

#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "EMCALWorkflow/CellConverterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsEMCAL/MCLabel.h"
#include "EMCALBase/Geometry.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/CaloRawFitterGamma2.h"

using namespace o2::emcal::reco_workflow;

void CellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[EMCALCellConverter - init] Initialize converter " << (mPropagateMC ? "with" : "without") << " MC truth container";

  if (!mGeometry) {
    mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(223409);
  }
  if (!mGeometry) {
    LOG(ERROR) << "Failure accessing geometry";
  }

  auto fitmethod = ctx.options().get<std::string>("fitmethod");
  if (fitmethod == "standard") {
    LOG(INFO) << "Using standard raw fitter";
    mRawFitter = std::unique_ptr<o2::emcal::CaloRawFitter>(new o2::emcal::CaloRawFitterStandard);
  } else if (fitmethod == "gamma2") {
    mRawFitter = std::unique_ptr<o2::emcal::CaloRawFitter>(new o2::emcal::CaloRawFitterGamma2);
  }
  mRawFitter->setAmpCut(0.);
  mRawFitter->setL1Phase(0.);
}

void CellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALCellConverter - run] called";
  const double CONVADCGEV = 0.016; // Conversion from ADC counts to energy: E = 16 MeV / ADC

  mOutputCells.clear();
  mOutputTriggers.clear();
  auto digitsAll = ctx.inputs().get<gsl::span<o2::emcal::Digit>>("digits");
  auto triggers = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("triggers");
  LOG(DEBUG) << "[EMCALCellConverter - run]  Received " << digitsAll.size() << " digits from " << triggers.size() << " trigger ...";
  int currentstart = mOutputCells.size(), ncellsTrigger = 0;
  for (const auto& trg : triggers) {
    if (!trg.getNumberOfObjects()) {
      mOutputTriggers.emplace_back(trg.getBCData(), currentstart, ncellsTrigger);
      continue;
    }
    gsl::span<const o2::emcal::Digit> digits(digitsAll.data() + trg.getFirstEntry(), trg.getNumberOfObjects());

    for (const auto& srucont : digitsToBunches(digits)) {

      if (srucont.mSRUid == 21 || srucont.mSRUid == 22 || srucont.mSRUid == 36 || srucont.mSRUid == 39) {
        continue;
      }

      for (const auto& [tower, channelBunches] : srucont.mChannelsBunches) {

        // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
        o2::emcal::CaloFitResults fitResults = mRawFitter->evaluate(channelBunches, 0, 0);
        if (fitResults.getAmp() < 0) {
          fitResults.setAmp(0.);
        }
        if (fitResults.getTime() < 0) {
          fitResults.setTime(0.);
        }

        mOutputCells.emplace_back(tower, fitResults.getAmp() * CONVADCGEV, fitResults.getTime(), ChannelType_t::HIGH_GAIN);
        ncellsTrigger++;
      }
    }
    mOutputTriggers.emplace_back(trg.getBCData(), currentstart, ncellsTrigger);
    currentstart = mOutputCells.size();
    ncellsTrigger = 0;
  }
  LOG(DEBUG) << "[EMCALCellConverter - run] Writing " << mOutputCells.size() << " cells ...";
  ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe}, mOutputCells);
  ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggers);
  if (mPropagateMC) {
    // copy mc truth container without modification
    // as indexing doesn't change
    auto truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>*>("digitsmctr");
    ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe}, *truthcont);
  }
}

std::vector<o2::emcal::SRUBunchContainer> CellConverterSpec::digitsToBunches(gsl::span<const o2::emcal::Digit> digits)
{

  std::vector<o2::emcal::SRUBunchContainer> sruBunchContainer;
  std::vector<o2::emcal::SRUDigitContainer> sruDigitContainer;

  for (auto iddl = 0; iddl < 40; iddl++) {
    o2::emcal::SRUBunchContainer srucontBunch;
    srucontBunch.mSRUid = iddl;
    sruBunchContainer.push_back(srucontBunch);

    o2::emcal::SRUDigitContainer srucontDigits;
    srucontDigits.mSRUid = iddl;
    sruDigitContainer.push_back(srucontDigits);
  }

  std::vector<const o2::emcal::Digit*>* bunchDigits;
  int lasttower = -1;
  for (auto& dig : digits) {
    auto tower = dig.getTower();
    if (tower != lasttower) {
      lasttower = tower;
      if (tower > 20000) {
        std::cout << "Wrong cell ID " << tower << std::endl;
      }

      auto onlineindices = mGeometry->getOnlineID(tower);
      int sruID = std::get<0>(onlineindices);

      auto towerdata = sruDigitContainer[sruID].mChannelsDigits.find(tower);
      if (towerdata == sruDigitContainer[sruID].mChannelsDigits.end()) {
        sruDigitContainer[sruID].mChannelsDigits[tower] = std::vector<const o2::emcal::Digit*>(o2::emcal::constants::EMCAL_MAXTIMEBINS);
        bunchDigits = &(sruDigitContainer[sruID].mChannelsDigits[tower]);
        memset(bunchDigits->data(), 0, sizeof(o2::emcal::Digit*) * o2::emcal::constants::EMCAL_MAXTIMEBINS);
      } else {
        bunchDigits = &(towerdata->second);
      }
    }

    // Get time sample of the digit:
    // Digitizer stores the time sample in ns, needs to be converted to time sample dividing
    // by the length of the time sample
    auto timesample = int(dig.getTimeStamp() / emcal::constants::EMCAL_TIMESAMPLE);
    if (timesample >= o2::emcal::constants::EMCAL_MAXTIMEBINS) {
      LOG(ERROR) << "Digit time sample " << timesample << " outside range [0," << o2::emcal::constants::EMCAL_MAXTIMEBINS << "]";
      continue;
    }
    (*bunchDigits)[timesample] = &dig;
  }

  for (auto srucont : sruDigitContainer) {

    if (srucont.mSRUid == 21 || srucont.mSRUid == 22 || srucont.mSRUid == 36 || srucont.mSRUid == 39) {
      continue;
    }

    for (const auto& [tower, channelDigits] : srucont.mChannelsDigits) {

      std::vector<o2::emcal::Bunch> rawbunches;
      for (auto& bunch : findBunches(channelDigits)) {
        rawbunches.emplace_back(bunch.mADCs.size(), bunch.mStarttime);
        for (auto adc : bunch.mADCs) {
          rawbunches.back().addADC(adc);
        }
      }

      sruBunchContainer[srucont.mSRUid].mChannelsBunches[tower] = rawbunches;
    }
  }

  return sruBunchContainer;
}

std::vector<o2::emcal::AltroBunch> CellConverterSpec::findBunches(const std::vector<const o2::emcal::Digit*>& channelDigits)
{
  std::vector<o2::emcal::AltroBunch> result;
  o2::emcal::AltroBunch* currentBunch = nullptr;
  // Digits in ALTRO bunch in time-reversed order
  int itime;
  for (itime = channelDigits.size() - 1; itime >= 0; itime--) {
    auto dig = channelDigits[itime];
    if (!dig) {
      if (currentBunch) {
        currentBunch->mStarttime = itime + 1;
        currentBunch = nullptr;
      }
      continue;
    }
    int adc = dig->getAmplitudeADC();
    // Apply pedestal at 3 ADC, @TODO Needs to come from CCDB
    if (adc < 3) {
      // Stop bunch
      // Set the start time to the time sample of previous (passing) digit
      currentBunch->mStarttime = itime + 1;
      currentBunch = nullptr;
      continue;
    }
    if (!currentBunch) {
      // start new bunch
      o2::emcal::AltroBunch bunch;
      result.push_back(bunch);
      currentBunch = &(result.back());
    }
    currentBunch->mADCs.emplace_back(adc);
  }
  // if we have a last bunch set time start time to the time bin of teh previous digit
  if (currentBunch) {
    currentBunch->mStarttime = itime + 1;
  }
  return result;
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getCellConverterSpec(bool propagateMC)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginEMC, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("triggers", "EMC", "DIGITSTRGR", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("EMC", "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("digitsmctr", "EMC", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("EMC", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe);
  }
  return o2::framework::DataProcessorSpec{"EMCALCellConverterSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::emcal::reco_workflow::CellConverterSpec>(propagateMC),
                                          o2::framework::Options{
                                            {"fitmethod", o2::framework::VariantType::String, "standard", {"Fit method (standard or gamma2)"}}}};
}
