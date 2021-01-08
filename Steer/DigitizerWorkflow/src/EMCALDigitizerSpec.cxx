// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALDigitizerSpec.h"
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
    LOG(ERROR) << "Geometry needs to be loaded before";
  }
  // run 3 geometry == run 2 geometry for EMCAL
  // to be adapted with run numbers at a later stage
  auto geom = o2::emcal::Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", "Geant4", "EMV-EMCAL");
  // init digitizer
  mDigitizer.setGeometry(geom);
  mDigitizer.init();

  mFinished = false;
}

void DigitizerSpec::run(framework::ProcessingContext& ctx)
{
  if (mFinished) {
    return;
  }

  // read collision context from input
  auto context = ctx.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
  context->initSimChains(o2::detectors::DetID::EMC, mSimChains);
  auto& timesview = context->getEventRecords();
  LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";

  // if there is nothing to do ... return
  if (timesview.size() == 0) {
    return;
  }

  TStopwatch timer;
  timer.Start();

  LOG(INFO) << " CALLING EMCAL DIGITIZATION ";
  o2::dataformats::MCTruthContainer<o2::emcal::MCLabel> labelAccum;
  std::vector<TriggerRecord> triggers;

  auto& eventParts = context->getEventParts();
  mDigitizer.clear();
  mAccumulatedDigits.clear();
  int trigID = 0;
  int indexStart = mAccumulatedDigits.size();
  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < timesview.size(); ++collID) {
    if (!mDigitizer.isEmpty() && (o2::emcal::SimParam::Instance().isDisablePileup() || !mDigitizer.isLive(timesview[collID].getTimeNS()))) {
      // copy digits into accumulator
      mDigits.clear();
      mLabels.clear();
      mDigitizer.fillOutputContainer(mDigits, mLabels);
      std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(mAccumulatedDigits));
      labelAccum.mergeAtBack(mLabels);
      LOG(INFO) << "Have " << mAccumulatedDigits.size() << " digits ";
      triggers.emplace_back(timesview[trigID], indexStart, mDigits.size());
      indexStart = mAccumulatedDigits.size();
      mDigits.clear();
      mLabels.clear();
    }

    mDigitizer.setEventTime(timesview[collID].getTimeNS());

    if (!mDigitizer.isLive()) {
      continue;
    }

    if (mDigitizer.isEmpty()) {
      mDigitizer.initCycle();
      trigID = collID;
    }

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {
      mDigitizer.setCurrEvID(part.entryID);
      mDigitizer.setCurrSrcID(part.sourceID);

      // get the hits for this event and this source
      mHits.clear();
      context->retrieveHits(mSimChains, "EMCHit", part.sourceID, part.entryID, &mHits);

      LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";

      // call actual digitization procedure
      mDigitizer.process(mHits);
    }
  }

  if (!mDigitizer.isEmpty()) {
    // copy digits into accumulator
    mDigits.clear();
    mLabels.clear();
    mDigitizer.fillOutputContainer(mDigits, mLabels);
    std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(mAccumulatedDigits));
    labelAccum.mergeAtBack(mLabels);
    LOG(INFO) << "Have " << mAccumulatedDigits.size() << " digits ";
    triggers.emplace_back(timesview[trigID], o2::trigger::PhT, indexStart, mDigits.size());
    indexStart = mAccumulatedDigits.size();
    mDigits.clear();
    mLabels.clear();
  }

  LOG(INFO) << "Have " << labelAccum.getNElements() << " EMCAL labels ";
  // here we have all digits and we can send them to consumer (aka snapshot it onto output)
  ctx.outputs().snapshot(Output{"EMC", "DIGITS", 0, Lifetime::Timeframe}, mAccumulatedDigits);
  ctx.outputs().snapshot(Output{"EMC", "TRGRDIG", 0, Lifetime::Timeframe}, triggers);
  if (ctx.outputs().isAllowed({"EMC", "DIGITSMCTR", 0})) {
    ctx.outputs().snapshot(Output{"EMC", "DIGITSMCTR", 0, Lifetime::Timeframe}, labelAccum);
  }
  // EMCAL is always a triggering detector
  const o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::TRIGGERING;
  LOG(INFO) << "EMCAL: Sending ROMode= " << roMode << " to GRPUpdater";
  ctx.outputs().snapshot(Output{"EMC", "ROMode", 0, Lifetime::Timeframe}, roMode);

  timer.Stop();
  LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";
  // we should be only called once; tell DPL that this process is ready to exit
  ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  mFinished = true;
}

DataProcessorSpec getEMCALDigitizerSpec(int channel, bool mctruth)
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
