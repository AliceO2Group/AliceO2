// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FairLogger.h"

#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "EMCALWorkflow/CellConverterSpec.h"
#include "Framework/ControlService.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::emcal::reco_workflow;

void CellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[EMCALCellConverter - init] Initialize converter " << (mPropagateMC ? "with" : "without") << " MC truth container";
}

void CellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALCellConverter - run] called";
  auto dataref = ctx.inputs().get("digits");
  auto const* emcheader = o2::framework::DataRefUtils::getHeader<o2::emcal::EMCALBlockHeader*>(dataref);
  if (!emcheader->mHasPayload) {
    LOG(DEBUG) << "[EMCALCellConverter - run] No more digits" << std::endl;
    ctx.services().get<o2::framework::ControlService>().readyToQuit(false);
    return;
  }

  mOutputCells.clear();
  auto digits = ctx.inputs().get<std::vector<o2::emcal::Digit>>("digits");
  LOG(DEBUG) << "[EMCALCellConverter - run]  Received " << digits.size() << " digits ...";
  for (const auto& dig : digits) {
    ChannelType_t chantype;
    if (dig.getHighGain())
      chantype = ChannelType_t::HIGH_GAIN;
    else if (dig.getLowGain())
      chantype = ChannelType_t::LOW_GAIN;
    else if (dig.getTRU())
      chantype = ChannelType_t::TRU;
    else if (dig.getLEDMon())
      chantype = ChannelType_t::LEDMON;
    mOutputCells.emplace_back(dig.getTower(), dig.getEnergy(), dig.getTimeStamp(), chantype);
  }
  LOG(DEBUG) << "[EMCALCellConverter - run] Writing " << mOutputCells.size() << " cells ...";
  ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe}, mOutputCells);
  if (mPropagateMC) {
    // copy mc truth container without modification
    // as indexing doesn't change
    auto truthcont = ctx.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("digitsmctr");
    ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe}, *truthcont);
  }
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getCellConverterSpec(bool propagateMC)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;
  inputs.emplace_back("digits", o2::header::gDataOriginEMC, "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe);
  if (propagateMC) {
    inputs.emplace_back("digitsmctr", "EMC", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("EMC", "CELLSMCTR", 0, o2::framework::Lifetime::Timeframe);
  }
  return o2::framework::DataProcessorSpec{"EMCALCellConverterSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::emcal::reco_workflow::CellConverterSpec>(propagateMC)};
}