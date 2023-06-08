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

#include "EMCALWorkflow/StandaloneAODProducerSpec.h"
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
namespace emcal
{

StandaloneAODProducerSpec::StandaloneAODProducerSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void StandaloneAODProducerSpec::init(o2::framework::InitContext& ic)
{
  mCaloEventHandler = new o2::emcal::EventHandler<o2::emcal::Cell>();
  mCaloAmp = 0xFFFFFFFF;
  mCaloTime = 0xFFFFFFFF;

  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mRunNumber = ic.options().get<int>("run-number");
}

void StandaloneAODProducerSpec::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);

  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  uint64_t tfNumber;
  if (mTFNumber == -1L) {
    // TODO has to be made globally unique (by using absolute time of TF). For now is unique within the run
    tfNumber = uint64_t(tinfo.firstTForbit) + (uint64_t(tinfo.runNumber) << 32); // getTFNumber(mStartIR, runNumber);
  } else {
    tfNumber = mTFNumber;
  }
  const int runNumber = (mRunNumber == -1) ? int(tinfo.runNumber) : mRunNumber;

  auto cellsIn = pc.inputs().get<gsl::span<o2::emcal::Cell>>(getCellBinding());
  auto triggersIn = pc.inputs().get<gsl::span<o2::emcal::TriggerRecord>>(getCellTriggerRecordBinding());

  LOG(info) << "FOUND " << cellsIn.size() << " EMC cells in CTF";
  LOG(info) << "FOUND " << triggersIn.size() << " EMC tiggers in CTF";

  auto bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto collisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "COLLISION"});
  auto caloCellsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALO"});
  auto caloCellsTRGTableBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "CALOTRIGGER"});

  auto bcCursor = bcBuilder->cursor<o2::aod::BCs>();
  auto collisionsCursor = collisionsBuilder->cursor<o2::aod::Collisions>();
  auto caloCellsCursor = caloCellsBuilder->cursor<o2::aod::Calos>();
  auto caloCellsTRGTableCursor = caloCellsTRGTableBuilder->cursor<o2::aod::CaloTriggers>();

  // build event for easier handling
  mCaloEventHandler->reset();
  mCaloEventHandler->setCellData(cellsIn, triggersIn);

  uint64_t triggerMask = 1;
  // loop over events
  for (int iev = 0; iev < mCaloEventHandler->getNumberOfEvents(); iev++) {
    o2::emcal::EventData inputEvent = mCaloEventHandler->buildEvent(iev);
    auto cellsInEvent = inputEvent.mCells;                  // get cells belonging to current event
    auto interactionRecord = inputEvent.mInteractionRecord; // get interaction records belonging to current event

    LOG(info) << "Found " << cellsInEvent.size() << " cells in event";

    auto bcID = interactionRecord.toLong();
    bcCursor(0,
             runNumber,
             bcID,
             triggerMask);
    auto indexBC = iev;

    for (auto& cell : cellsInEvent) {

      // fill table
      caloCellsCursor(0,
                      indexBC,
                      cell.getTower(),
                      o2::math_utils::detail::truncateFloatFraction(cell.getAmplitude(), mCaloAmp),
                      o2::math_utils::detail::truncateFloatFraction(cell.getTimeStamp(), mCaloTime),
                      cell.getType(),
                      1); // hard coded for emcal (-1 would be undefined, 0 phos)
    }                     // end of cell loop

    // filled only with BCID, rest dummy for no2
    caloCellsTRGTableCursor(0,
                            indexBC,
                            0,  // fastOrAbsId (dummy value)
                            0., // lnAmplitude (dummy value)
                            0,  // triggerBits (dummy value)
                            1); // caloType (dummy value)

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
  LOGF(info, "EMCAL Standalone AOD Producer total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getStandaloneAODProducerSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2collision"}, "AOD", "COLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCell"}, "AOD", "CALO", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2caloCellTRGR"}, "AOD", "CALOTRIGGER", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});

  return DataProcessorSpec{
    "standalone-aod-producer-workflow",
    Inputs{
      InputSpec{StandaloneAODProducerSpec::getCellTriggerRecordBinding(), "EMC", "CELLSTRGR", 0, Lifetime::Timeframe},
      InputSpec{StandaloneAODProducerSpec::getCellBinding(), "EMC", "CELLS", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<StandaloneAODProducerSpec>()},
    Options{
      ConfigParamSpec{"run-number", VariantType::Int64, -1L, {"The run-number. If left default we try to get it from DPL header."}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}}}};
}

} // namespace emcal
} // namespace o2
