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

#include "EMCALWorkflow/EMCALDigitizerSpec.h"
#include "CommonConstants/Triggers.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <TGeoManager.h>

#include "CommonDataFormat/EvIndex.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsEMCAL/TriggerRecord.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace emcal
{

void DigitizerSpec::initDigitizerTask(framework::InitContext& ctx)
{
  if (!gGeoManager) {
    LOG(error) << "Geometry needs to be loaded before";
  }
  // run 3 geometry == run 2 geometry for EMCAL
  // to be adapted with run numbers at a later stage
  auto geom = o2::emcal::Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", "Geant4", "EMV-EMCAL");
  // init digitizer

  mSumDigitizer.setGeometry(geom);

  mDigitizer.init();

  mFinished = false;
}

void DigitizerSpec::run(framework::ProcessingContext& ctx)
{
  if (mFinished) {
    return;
  }
  mDigitizer.flush();

  // read collision context from input
  auto context = ctx.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
  context->initSimChains(o2::detectors::DetID::EMC, mSimChains);
  auto& timesview = context->getEventRecords();
  LOG(debug) << "GOT " << timesview.size() << " COLLISSION TIMES";

  // if there is nothing to do ... return
  if (timesview.size() == 0) {
    return;
  }

  TStopwatch timer;
  timer.Start();

  LOG(info) << " CALLING EMCAL DIGITIZATION ";
  o2::dataformats::MCTruthContainer<o2::emcal::MCLabel> labelAccum;

  auto& eventParts = context->getEventParts();

  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < timesview.size(); ++collID) {

    mDigitizer.setEventTime(timesview[collID]);

    if (!mDigitizer.isLive()) {
      continue;
    }

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {

      mSumDigitizer.setCurrEvID(part.entryID);
      mSumDigitizer.setCurrSrcID(part.sourceID);

      // get the hits for this event and this source
      mHits.clear();
      context->retrieveHits(mSimChains, "EMCHit", part.sourceID, part.entryID, &mHits);

      LOG(info) << "For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";

      std::vector<o2::emcal::LabeledDigit> summedDigits = mSumDigitizer.process(mHits);

      // call actual digitization procedure
      mDigitizer.process(summedDigits);
    }
  }

  mDigitizer.finish();

  // here we have all digits and we can send them to consumer (aka snapshot it onto output)
  ctx.outputs().snapshot(Output{"EMC", "DIGITS", 0, Lifetime::Timeframe}, mDigitizer.getDigits());
  ctx.outputs().snapshot(Output{"EMC", "TRGRDIG", 0, Lifetime::Timeframe}, mDigitizer.getTriggerRecords());
  if (ctx.outputs().isAllowed({"EMC", "DIGITSMCTR", 0})) {
    ctx.outputs().snapshot(Output{"EMC", "DIGITSMCTR", 0, Lifetime::Timeframe}, mDigitizer.getMCLabels());
  }
  // EMCAL is always a triggering detector
  const o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::TRIGGERING;
  LOG(info) << "EMCAL: Sending ROMode= " << roMode << " to GRPUpdater";
  ctx.outputs().snapshot(Output{"EMC", "ROMode", 0, Lifetime::Timeframe}, roMode);

  timer.Stop();
  LOG(info) << "Digitization took " << timer.CpuTime() << "s";
  // we should be only called once; tell DPL that this process is ready to exit
  ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  mFinished = true;
}

o2::framework::DataProcessorSpec getEMCALDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("EMC", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("EMC", "TRGRDIG", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("EMC", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("EMC", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "EMCALDigitizer", Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{o2::framework::adaptFromTask<DigitizerSpec>()},
    Options{{"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}
    // I can't use VariantType::Bool as it seems to have a problem
  };
}
} // end namespace emcal
} // end namespace o2
