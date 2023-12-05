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
#include <fairlogger/Logger.h>

#include "Framework/RootSerializationSupport.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "PHOSWorkflow/CellConverterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "PHOSBase/PHOSSimParams.h"

using namespace o2::phos::reco_workflow;

void CellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(info) << "[PHOSCellConverter - init] Initialize converter " << (mPropagateMC ? "with" : "without") << " MC truth container";
  if (mDefBadMap) {
    LOG(info) << "No reading BadMap from ccdb requested, set default";
    // create test BadMap and Calib objects. ClusterizerSpec should be owner
    mBadMap = std::make_unique<BadChannelsMap>(); // Create empty bad map
    mHasCalib = true;
  }
}

void CellConverterSpec::run(framework::ProcessingContext& ctx)
{
  // LOG(debug) << "[PHOSCellConverter - run] called";
  // auto dataref = ctx.inputs().get("digits");
  // auto const* phosheader = o2::framework::DataRefUtils::getHeader<o2::phos::PHOSBlockHeader*>(dataref);
  // if (!phosheader->mHasPayload) {
  auto digitsTR = ctx.inputs().get<std::vector<o2::phos::TriggerRecord>>("digitTriggerRecords");
  if (!digitsTR.size()) { // nothing to process
    mOutputCells.clear();
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLS", 0}, mOutputCells);
    mOutputCellTrigRecs.clear();
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLTRIGREC", 0}, mOutputCellTrigRecs);
    if (mPropagateMC) {
      mOutputTruthCont.clear();
      ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLSMCTR", 0}, mOutputTruthCont);
    }
    return;
  }

  if (mInitSimParams) { // trigger reading sim/rec parameters from CCDB, singleton initiated in Fetcher
    ctx.inputs().get<o2::phos::PHOSSimParams*>("recoparams");
    mInitSimParams = false;
  }

  mOutputCells.clear();
  mOutputCellTrigRecs.clear();

  auto digits = ctx.inputs().get<std::vector<o2::phos::Digit>>("digits");
  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::phos::MCLabel>> truthcont(nullptr);
  if (mPropagateMC) {
    truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>*>("digitsmctr");
    mOutputTruthCont.clear();
  }
  if (mPropagateMC) {
    LOG(info) << "[PHOSCellConverter - run]  Received " << digits.size() << " digits and " << digitsTR.size() << " TriggerRecords" << truthcont->getNElements() << " MC labels";
  } else {
    LOG(info) << "[PHOSCellConverter - run]  Received " << digits.size() << " digits and " << digitsTR.size() << " TriggerRecords";
  }

  // get BadMap from CCDB, once
  if (!mHasCalib) {
    auto badMapPtr = ctx.inputs().get<o2::phos::BadChannelsMap*>("badmap");
    mBadMap = std::make_unique<BadChannelsMap>(*(badMapPtr.get()));
    mHasCalib = true;
  }

  mOutputCells.reserve(digits.size()); // most of digits will be copied
  int icell = 0;
  for (const auto& tr : digitsTR) {
    int iFirstDigit = tr.getFirstEntry();
    int iLastDigit = iFirstDigit + tr.getNumberOfObjects();
    int indexStart = mOutputCells.size();
    for (int i = iFirstDigit; i < iLastDigit; i++) {
      const auto& dig = digits.at(i);

      if (dig.isTRU()) {
        ChannelType_t chantype;
        if (dig.isHighGain()) {
          chantype = ChannelType_t::TRU2x2;
        } else {
          chantype = ChannelType_t::TRU4x4;
        }
        mOutputCells.emplace_back(dig.getAbsId(), dig.getAmplitude(), dig.getTime(), chantype);
      } else {
        // apply filter
        if (!mBadMap->isChannelGood(dig.getAbsId())) {
          continue;
        }

        ChannelType_t chantype;
        if (dig.isHighGain()) {
          chantype = ChannelType_t::HIGH_GAIN;
        } else {
          chantype = ChannelType_t::LOW_GAIN;
        }
        mOutputCells.emplace_back(dig.getAbsId(), dig.getAmplitude(), dig.getTime(), chantype);
        if (mPropagateMC) { // copy MC info,
          int iLab = dig.getLabel();
          if (iLab > -1) {
            mOutputTruthCont.addElements(icell, truthcont->getLabels(iLab));
          } else {
            MCLabel label(0, 0, 0, true, 0);
            label.setNoise();
            mOutputTruthCont.addElement(icell, label);
          }
          icell++;
        }
      }
    }
    mOutputCellTrigRecs.emplace_back(tr.getBCData(), indexStart, mOutputCells.size() - indexStart);
  }
  LOG(info) << "[PHOSCellConverter - run] Writing " << mOutputCells.size() << " cells, " << mOutputCellTrigRecs.size() << " Trig Records " << mOutputTruthCont.getNElements() << " PHOS labels ";

  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLS", 0}, mOutputCells);
  ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLTRIGREC", 0}, mOutputCellTrigRecs);
  if (mPropagateMC) {
    ctx.outputs().snapshot(o2::framework::Output{"PHS", "CELLSMCTR", 0}, mOutputTruthCont);
  }
}

o2::framework::DataProcessorSpec o2::phos::reco_workflow::getCellConverterSpec(bool propagateMC, bool defBadMap)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginPHS, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("digitTriggerRecords", o2::header::gDataOriginPHS, "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (!defBadMap) {
    inputs.emplace_back("badmap", o2::header::gDataOriginPHS, "PHS_Calib_BadMap", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/BadMap"));
  }
  inputs.emplace_back("recoparams", o2::header::gDataOriginPHS, "PHS_RecoParams", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Config/RecoParams"));

  outputs.emplace_back("PHS", "CELLS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("PHS", "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("digitsmctr", "PHS", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("PHS", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe);
  }
  return o2::framework::DataProcessorSpec{"PHOSCellConverterSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::phos::reco_workflow::CellConverterSpec>(propagateMC, defBadMap)};
}
