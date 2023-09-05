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

#include "PHOSWorkflow/StandaloneAODProducerSpec.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/Cell.h"
#include "Framework/TableBuilder.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "MathUtils/Utils.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

StandaloneAODProducerSpec::StandaloneAODProducerSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void StandaloneAODProducerSpec::init(o2::framework::InitContext& ic)
{
  mCaloAmp = 0xFFFFFFFF;
  mCaloTime = 0xFFFFFFFF;

  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mRunNumber = ic.options().get<int>("run-number");
}

void StandaloneAODProducerSpec::run(ProcessingContext& pc)
{
  // auto cput = mTimer.CpuTime();
  // mTimer.Start(false);
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  uint64_t tfNumber;
  if (mTFNumber == -1L) {
    // TODO has to be made globally unique (by using absolute time of TF). For now is unique within the run
    tfNumber = uint64_t(tinfo.firstTForbit) + (uint64_t(tinfo.runNumber) << 32); // getTFNumber(mStartIR, runNumber);
  } else {
    tfNumber = mTFNumber;
  }
  const int runNumber = (mRunNumber == -1) ? int(tinfo.runNumber) : mRunNumber;

  auto cells = pc.inputs().get<gsl::span<o2::phos::Cell>>("cells");
  auto ctr = pc.inputs().get<gsl::span<o2::phos::TriggerRecord>>("cellTriggerRecords");

  LOG(debug) << "FOUND " << cells.size() << " PHOS cells in CTF";
  LOG(debug) << "FOUND " << ctr.size() << " PHOS tiggers in CTF";

  auto bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
  auto caloCellsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALO"});
  auto caloCellsTRGTableBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALOTRIGGER"});

  auto bcCursor = bcBuilder->cursor<o2::aod::BCs>();
  auto collisionsCursor = collisionsBuilder->cursor<o2::aod::Collisions>();
  auto caloCellsCursor = caloCellsBuilder->cursor<o2::aod::Calos>();
  auto caloCellsTRGTableCursor = caloCellsTRGTableBuilder->cursor<o2::aod::CaloTriggers>();

  uint64_t triggerMask = 1, inputMask = 1;
  // loop over events
  int indexBC = -1;
  for (const auto& tr : ctr) {
    indexBC++;
    int firstCellInEvent = tr.getFirstEntry();
    int lastCellInEvent = firstCellInEvent + tr.getNumberOfObjects();
    for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
      const Cell c = cells[i];
      if (c.getTRU()) {
        // TODO!!! test time?
        caloCellsTRGTableCursor(0,
                                indexBC,
                                c.getTRUId(),                    // fastOrAbsId
                                c.getEnergy(),                   // lnAmplitude (dummy value)
                                (c.getType() == TRU2x2) ? 0 : 1, // triggerBits 0:L0 2x2, 1: L1 4x4
                                0);                              // caloType 0: PHOS
      }

      // TODO: should bad map be applied here? Unrecoverable loss of channels: special loose map?
      // short absId = c.getAbsId();
      // if (isBadChannel(absId)) {
      //   continue;
      // }
      // fill table
      caloCellsCursor(0,
                      indexBC,
                      c.getAbsId(),
                      o2::math_utils::detail::truncateFloatFraction(c.getEnergy(), mCaloAmp),
                      o2::math_utils::detail::truncateFloatFraction(c.getTime(), mCaloTime),
                      c.getType(), // HG/LG
                      0);          // hard coded for phos (-1 would be undefined, 0 phos)
    }                              // end of cell loop

    auto bcID = tr.getBCData().toLong();
    bcCursor(0,
             runNumber,
             bcID,
             triggerMask,
             inputMask);

    // fill collision cursor
    collisionsCursor(0,
                     indexBC,
                     0., // X-Pos dummy value
                     0., // Y Pos
                     0., // Z Pos
                     0,  // cov 0
                     0,  // cov 1
                     0,  // cov 2
                     0,  // cov 3
                     0,  // cov 4
                     0,  // cov 5
                     0,  // vertex bit field for flags
                     0,  // chi2
                     0,  // ncontributors
                     0,  // rel interaction time
                     0); // vertex time stamp

  } // end of event loop
  // std::cout << "Finished cell loop" << std::endl;

  pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, tfNumber);
  pc.outputs().snapshot(Output{"TFF", "TFFilename", 0, Lifetime::Timeframe}, "");

  mTimer.Stop();
}

void StandaloneAODProducerSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "PHOS Standalone AOD Producer total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getPHOSStandaloneAODProducerSpec()
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("cells", o2::header::gDataOriginPHS, "CELLS", 0, Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", o2::header::gDataOriginPHS, "CELLTRIGREC", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCell"}, "AOD", "CALO", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCellTRGR"}, "AOD", "CALOTRIGGER", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});

  return DataProcessorSpec{
    "phos-standalone-aod-producer-workflow",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<StandaloneAODProducerSpec>()},
    Options{
      ConfigParamSpec{"run-number", VariantType::Int64, -1L, {"The run-number. If left default we try to get it from DPL header."}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}}}};
}

} // namespace phos
} // namespace o2
