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
#include "DataFormatsEMCAL/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::emcal::reco_workflow;

void CellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[EMCALCellConverter - init] Initialize converter " << (mPropagateMC ? "with" : "without") << " MC truth container";
}

void CellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALCellConverter - run] called";
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
    for (const auto& dig : digits) {
      ChannelType_t chantype;
      if (dig.getHighGain()) {
        chantype = ChannelType_t::HIGH_GAIN;
      } else if (dig.getLowGain()) {
        chantype = ChannelType_t::LOW_GAIN;
      } else if (dig.getTRU()) {
        chantype = ChannelType_t::TRU;
      } else if (dig.getLEDMon()) {
        chantype = ChannelType_t::LEDMON;
      }
      mOutputCells.emplace_back(dig.getTower(), dig.getEnergy(), dig.getTimeStamp(), chantype);
      ncellsTrigger++;
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
                                          o2::framework::adaptFromTask<o2::emcal::reco_workflow::CellConverterSpec>(propagateMC)};
}
