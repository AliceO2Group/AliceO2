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
#include <gsl/span>
#include <boost/range/combine.hpp>
#include "FairLogger.h"

#include "DataFormatsEMCAL/Digit.h"
#include "EMCALWorkflow/CellConverterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsEMCAL/MCLabel.h"
#include "EMCALBase/Geometry.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/CaloRawFitterGamma2.h"
#include "EMCALReconstruction/RecoParam.h"

using namespace o2::emcal::reco_workflow;

void CellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(debug) << "[EMCALCellConverter - init] Initialize converter " << (mPropagateMC ? "with" : "without") << " MC truth container";

  if (!mGeometry) {
    mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(223409);
  }
  if (!mGeometry) {
    LOG(error) << "Failure accessing geometry";
  }

  auto fitmethod = ctx.options().get<std::string>("fitmethod");
  if (fitmethod == "standard") {
    LOG(info) << "Using standard raw fitter";
    mRawFitter = std::unique_ptr<o2::emcal::CaloRawFitter>(new o2::emcal::CaloRawFitterStandard);
  } else if (fitmethod == "gamma2") {
    LOG(info) << "Using gamma2 raw fitter";
    mRawFitter = std::unique_ptr<o2::emcal::CaloRawFitter>(new o2::emcal::CaloRawFitterGamma2);
  }
  mRawFitter->setAmpCut(0.);
  mRawFitter->setL1Phase(0.);
  LOG(info) << "Using time shift: " << RecoParam::Instance().getCellTimeShiftNanoSec() << " ns";
}

void CellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(debug) << "[EMCALCellConverter - run] called";
  double timeshift = RecoParam::Instance().getCellTimeShiftNanoSec(); // subtract offset in ns in order to center the time peak around the nominal delay

  mOutputCells.clear();
  mOutputLabels.clear();
  mOutputTriggers.clear();
  auto digitsAll = ctx.inputs().get<gsl::span<o2::emcal::Digit>>("digits");
  auto triggers = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("triggers");
  auto truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>*>("digitsmctr");

  LOG(debug) << "[EMCALCellConverter - run]  Received " << digitsAll.size() << " digits from " << triggers.size() << " trigger ...";
  int currentstart = mOutputCells.size(), ncellsTrigger = 0;
  for (const auto& trg : triggers) {
    if (!trg.getNumberOfObjects()) {
      mOutputTriggers.emplace_back(trg.getBCData(), trg.getTriggerBits(), currentstart, ncellsTrigger);
      continue;
    }

    gsl::span<const o2::emcal::Digit> digits(digitsAll.data() + trg.getFirstEntry(), trg.getNumberOfObjects());
    std::vector<gsl::span<const o2::emcal::MCLabel>> mcLabels;

    if (mPropagateMC) {
      for (int digitIndex = trg.getFirstEntry(); digitIndex < (trg.getFirstEntry() + trg.getNumberOfObjects()); digitIndex++) {
        mcLabels.push_back(truthcont->getLabels(digitIndex));
      }
    }

    for (const auto& srucont : digitsToBunches(digits, mcLabels)) {

      if (srucont.mSRUid == 21 || srucont.mSRUid == 22 || srucont.mSRUid == 36 || srucont.mSRUid == 39) {
        continue;
      }

      for (const auto& [tower, channelData] : srucont.mChannelsData) {

        // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
        ChannelType_t channelType = channelData.mChanType;
        CaloFitResults fitResults;
        try {
          fitResults = mRawFitter->evaluate(channelData.mChannelsBunchesHG);

          // If the high gain bunch is saturated then fit the low gain
          if (fitResults.getAmp() > o2::emcal::constants::OVERFLOWCUT) {
            fitResults = mRawFitter->evaluate(channelData.mChannelsBunchesLG);
            fitResults.setAmp(fitResults.getAmp() * o2::emcal::constants::EMCAL_HGLGFACTOR);
            channelType = ChannelType_t::LOW_GAIN;
          } else {
            channelType = ChannelType_t::HIGH_GAIN;
          }

          if (fitResults.getAmp() < 0) {
            fitResults.setAmp(0.);
          }
          if (fitResults.getTime() < 0) {
            fitResults.setTime(0.);
          }
          mOutputCells.emplace_back(tower, fitResults.getAmp() * o2::emcal::constants::EMCAL_ADCENERGY, fitResults.getTime() - timeshift, channelType);

          if (mPropagateMC) {
            Int_t LabelIndex = mOutputLabels.getIndexedSize();
            if (channelType == ChannelType_t::HIGH_GAIN) {
              // if this channel has no bunches, then fill an empty label
              if (channelData.mChannelLabelsHG.size() == 0) {
                const o2::emcal::MCLabel label = o2::emcal::MCLabel(false, 1.);
                mOutputLabels.addElementRandomAccess(LabelIndex, label);
              } else {
                // Fill only labels that corresponds to bunches with maximum ADC
                const int bunchindex = selectMaximumBunch(channelData.mChannelsBunchesHG);
                for (const auto& label : channelData.mChannelLabelsHG[bunchindex]) {
                  mOutputLabels.addElementRandomAccess(LabelIndex, label);
                }
              }
            } else {
              // if this channel has no bunches, then fill an empty label
              if (channelData.mChannelLabelsLG.size() == 0) {
                const o2::emcal::MCLabel label = o2::emcal::MCLabel(false, 1.);
                mOutputLabels.addElementRandomAccess(LabelIndex, label);
              } else {
                // Fill only labels that corresponds to bunches with maximum ADC
                const int bunchindex = selectMaximumBunch(channelData.mChannelsBunchesLG);
                for (const auto& label : channelData.mChannelLabelsLG[bunchindex]) {
                  mOutputLabels.addElementRandomAccess(LabelIndex, label);
                }
              }
            }
          }
          ncellsTrigger++;

        } catch (CaloRawFitter::RawFitterError_t& fiterror) {
          if (fiterror != CaloRawFitter::RawFitterError_t::BUNCH_NOT_OK) {
            LOG(error) << "Failure in raw fitting: " << CaloRawFitter::createErrorMessage(fiterror);
          }
        }
      }
    }
    mOutputTriggers.emplace_back(trg.getBCData(), trg.getTriggerBits(), currentstart, ncellsTrigger);
    currentstart = mOutputCells.size();
    ncellsTrigger = 0;
  }
  LOG(debug) << "[EMCALCellConverter - run] Writing " << mOutputCells.size() << " cells ...";
  ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe}, mOutputCells);
  ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggers);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe}, mOutputLabels);
  }
}

std::vector<o2::emcal::SRUBunchContainer> CellConverterSpec::digitsToBunches(gsl::span<const o2::emcal::Digit> digits, std::vector<gsl::span<const o2::emcal::MCLabel>>& mcLabels)
{

  std::vector<o2::emcal::SRUBunchContainer> sruBunchContainer;
  std::vector<o2::emcal::DigitContainerPerSRU> sruDigitContainer;

  for (auto iddl = 0; iddl < 40; iddl++) {
    o2::emcal::SRUBunchContainer srucontBunch;
    srucontBunch.mSRUid = iddl;
    sruBunchContainer.push_back(srucontBunch);

    o2::emcal::DigitContainerPerSRU srucontDigits;
    srucontDigits.mSRUid = iddl;
    sruDigitContainer.push_back(srucontDigits);
  }

  std::vector<const o2::emcal::Digit*>* bunchDigits;
  std::vector<gsl::span<const o2::emcal::MCLabel>>* bunchLabels;
  int lasttower = -1;
  // for (auto& dig : digits) {
  for (const auto& [dig, labels] : boost::combine(digits, mcLabels)) {
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
        sruDigitContainer[sruID].mChannelsDigits[tower] = {dig.getType(), std::vector<const o2::emcal::Digit*>(o2::emcal::constants::EMCAL_MAXTIMEBINS), mPropagateMC ? std::vector<gsl::span<const o2::emcal::MCLabel>>(o2::emcal::constants::EMCAL_MAXTIMEBINS) : std::vector<gsl::span<const o2::emcal::MCLabel>>()};
        bunchDigits = &(sruDigitContainer[sruID].mChannelsDigits[tower].mChannelDigits);
        memset(bunchDigits->data(), 0, sizeof(o2::emcal::Digit*) * o2::emcal::constants::EMCAL_MAXTIMEBINS);
        if (mPropagateMC) {
          bunchLabels = &(sruDigitContainer[sruID].mChannelsDigits[tower].mChannelLabels);
          memset(bunchLabels->data(), 0, sizeof(gsl::span<const o2::emcal::MCLabel>) * o2::emcal::constants::EMCAL_MAXTIMEBINS);
        }
      } else {
        bunchDigits = &(towerdata->second.mChannelDigits);
        if (mPropagateMC) {
          bunchLabels = &(towerdata->second.mChannelLabels);
        }
      }
    }

    // Get time sample of the digit:
    // Digitizer stores the time sample in ns, needs to be converted to time sample dividing
    // by the length of the time sample
    auto timesample = int(dig.getTimeStamp() / emcal::constants::EMCAL_TIMESAMPLE);
    if (timesample >= o2::emcal::constants::EMCAL_MAXTIMEBINS) {
      LOG(error) << "Digit time sample " << timesample << " outside range [0," << o2::emcal::constants::EMCAL_MAXTIMEBINS << "]";
      continue;
    }
    (*bunchDigits)[timesample] = &dig;
    if (mPropagateMC) {
      (*bunchLabels)[timesample] = labels;
    }
  }

  for (auto srucont : sruDigitContainer) {

    if (srucont.mSRUid == 21 || srucont.mSRUid == 22 || srucont.mSRUid == 36 || srucont.mSRUid == 39) {
      continue;
    }

    for (const auto& [tower, channelDigits] : srucont.mChannelsDigits) {

      std::vector<o2::emcal::Bunch> rawbunchesHG;
      std::vector<o2::emcal::Bunch> rawbunchesLG;
      std::vector<std::vector<o2::emcal::MCLabel>> rawLabelsHG;
      std::vector<std::vector<o2::emcal::MCLabel>> rawLabelsLG;

      bool saturatedBunch = false;

      // Creating the high gain bunch with labels
      for (auto& bunch : findBunches(channelDigits.mChannelDigits, channelDigits.mChannelLabels, ChannelType_t::HIGH_GAIN)) {
        rawbunchesHG.emplace_back(bunch.mADCs.size(), bunch.mStarttime);
        for (auto adc : bunch.mADCs) {
          rawbunchesHG.back().addADC(adc);
          if (adc > o2::emcal::constants::LG_SUPPRESSION_CUT) {
            saturatedBunch = true;
          }
        }
        if (mPropagateMC) {
          rawLabelsHG.push_back(bunch.mLabels);
        }
      }

      // Creating the low gain bunch with labels if the HG bunch is saturated
      if (saturatedBunch) {
        for (auto& bunch : findBunches(channelDigits.mChannelDigits, channelDigits.mChannelLabels, ChannelType_t::LOW_GAIN)) {
          rawbunchesLG.emplace_back(bunch.mADCs.size(), bunch.mStarttime);
          for (auto adc : bunch.mADCs) {
            rawbunchesLG.back().addADC(adc);
          }
          if (mPropagateMC) {
            rawLabelsLG.push_back(bunch.mLabels);
          }
        }
      }

      sruBunchContainer[srucont.mSRUid].mChannelsData[tower] = {channelDigits.mChanType, rawbunchesHG, rawbunchesLG, rawLabelsHG, rawLabelsLG};
    }
  }

  return sruBunchContainer;
}

std::vector<o2::emcal::AltroBunch> CellConverterSpec::findBunches(const std::vector<const o2::emcal::Digit*>& channelDigits, const std::vector<gsl::span<const o2::emcal::MCLabel>>& mcLabels, ChannelType_t channelType)
{
  std::vector<AltroBunch> result;
  AltroBunch currentBunch;
  bool bunchStarted = false;
  // Digits in ALTRO bunch in time-reversed order
  int itime;
  for (itime = channelDigits.size() - 1; itime >= 0; itime--) {
    auto dig = channelDigits[itime];
    auto labels = mcLabels[itime];
    if (!dig) {
      if (bunchStarted) {
        // we have a bunch which is started and needs to be closed
        // check if the ALTRO bunch has a minimum amount of 3 ADCs
        if (currentBunch.mADCs.size() >= 3) {
          // Bunch selected, set start time and push to bunches
          result.push_back(currentBunch);
          currentBunch = AltroBunch();
          bunchStarted = false;
        }
      }
      continue;
    }
    int adc = dig->getAmplitudeADC(channelType);
    if (adc < 3) {
      // ADC value below threshold
      // in case we have an open bunch it needs to be stopped bunch
      // Set the start time to the time sample of previous (passing) digit
      if (bunchStarted) {
        // check if the ALTRO bunch has a minimum amount of ADCs
        if (currentBunch.mADCs.size() >= 3) {
          // Bunch selected, set start time and push to bunches
          result.push_back(currentBunch);
          currentBunch = AltroBunch();
          bunchStarted = false;
        }
      }
    }
    // Valid ADC value, if the bunch is closed we start a new bunch
    if (!bunchStarted) {
      bunchStarted = true;
      currentBunch.mStarttime = itime;
    }
    currentBunch.mADCs.emplace_back(adc);
    if (mPropagateMC) {
      std::vector<o2::emcal::MCLabel> vectorLabels;
      for (auto& label : labels) {
        vectorLabels.push_back(label);
      }
      currentBunch.mEnergyLabels.insert(std::pair<int, std::vector<o2::emcal::MCLabel>>(adc, vectorLabels));
    }
  }
  // if we have a last bunch set time start time to the time bin of teh previous digit
  if (bunchStarted) {
    if (currentBunch.mADCs.size() >= 3) {
      result.push_back(currentBunch);
    }
  }

  if (mPropagateMC) {
    mergeLabels(result);
  }
  return result;
}

void CellConverterSpec::mergeLabels(std::vector<o2::emcal::AltroBunch>& channelBunches)
{
  for (auto& altroBunch : channelBunches) {
    std::vector<o2::emcal::MCLabel> mcLabels;
    std::map<uint64_t, std::vector<o2::emcal::MCLabel>> LabelsPerSource;
    double totalEnergy = 0;
    for (auto& [energy, Labels] : altroBunch.mEnergyLabels) {
      for (auto& trueLabel : Labels) {
        o2::emcal::MCLabel newLabel = trueLabel;
        newLabel.setAmplitudeFraction(trueLabel.getAmplitudeFraction() * energy);
        LabelsPerSource[newLabel.getRawValue()].push_back(newLabel);
      }
      totalEnergy += energy;
    }

    for (auto& [trackID, labels] : LabelsPerSource) {
      o2::emcal::MCLabel newLabel = labels[0];
      for (int ilabel = 0; ilabel < labels.size(); ilabel++) {
        auto& trueLabel = labels[ilabel];
        if (ilabel == 0) {
          continue;
        }
        newLabel.setAmplitudeFraction(newLabel.getAmplitudeFraction() + trueLabel.getAmplitudeFraction());
      }
      newLabel.setAmplitudeFraction(newLabel.getAmplitudeFraction() / totalEnergy);
      mcLabels.push_back(newLabel);
    }

    std::sort(mcLabels.begin(), mcLabels.end(), [](o2::emcal::MCLabel label1, o2::emcal::MCLabel label2) { return label1.getAmplitudeFraction() > label2.getAmplitudeFraction(); });

    altroBunch.mLabels = mcLabels;
    // std::copy(mcLabels.begin(), mcLabels.end(), std::back_inserter(altroBunch.mLabels));
    // altroBunch.mLabels = gsl::span<o2::emcal::MCLabel>(&mcLabels[0], mcLabels.size());
    altroBunch.mEnergyLabels.clear();
  }
}

int CellConverterSpec::selectMaximumBunch(const gsl::span<const Bunch>& bunchvector)
{
  int bunchindex = -1;
  int indexMaxInBunch(0), maxADCallBunches(-1);

  for (unsigned int i = 0; i < bunchvector.size(); i++) {
    auto [maxADC, maxIndex] = mRawFitter->getMaxAmplitudeBunch(bunchvector[i]); // CRAP PTH, bug fix, trouble if more than one bunches
    if (mRawFitter->isInTimeRange(maxIndex, mRawFitter->getMaxTimeIndex(), mRawFitter->getMinTimeIndex())) {
      if (maxADC > maxADCallBunches) {
        bunchindex = i;
        indexMaxInBunch = maxIndex;
        maxADCallBunches = maxADC;
      }
    }
  }

  return bunchindex;
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
                                            {"fitmethod", o2::framework::VariantType::String, "gamma2", {"Fit method (standard or gamma2)"}}}};
}
