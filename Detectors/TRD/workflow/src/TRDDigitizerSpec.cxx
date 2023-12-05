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

#include "TRDWorkflow/TRDDigitizerSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "TChain.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include "Framework/Task.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/Hit.h"
#include "DataFormatsTRD/Digit.h" // for the Digit type
#include "TRDBase/Calibrations.h"
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/Detector.h" // for the Hit type
#include "TRDSimulation/TRDSimParams.h"
#include <chrono>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace trd
{

class TRDDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
 public:
  TRDDPLDigitizerTask() : o2::base::BaseDPLDigitizer(o2::base::InitServices::GEOM | o2::base::InitServices::FIELD) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(info) << "initializing TRD digitization";
    mDigitizer.init();
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    LOG(info) << "Doing TRD digitization";

    bool mctruth = pc.outputs().isAllowed({"TRD", "LABELS", 0});

    Calibrations simcal;
    // the timestamp can be extracted from the DPL header (it is set in SimReader)
    auto creationTime = pc.services().get<o2::framework::TimingInfo>().creation;
    simcal.getCCDBObjects(creationTime);
    mDigitizer.setCalibrations(&simcal);

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(o2::detectors::DetID::TRD, mSimChains);
    auto& irecords = context->getEventRecords();

    for (auto& record : irecords) {
      LOG(debug) << "TRD TIME RECEIVED " << record.getTimeNS();
    }

    auto& eventParts = context->getEventParts();
    std::vector<o2::trd::Digit> digitsAccum; // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelsAccum;
    std::vector<TriggerRecord> triggers;

    std::vector<o2::trd::Digit> digits;                        // digits which get filled
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels; // labels which get filled

    o2::InteractionTimeRecord currentTime;  // the current time
    o2::InteractionTimeRecord previousTime; // the time of the previous collision
    o2::InteractionTimeRecord triggerTime;  // the time at which the TRD start reading out a signal
    size_t currTrig = 0;                    // from which collision is the current TRD trigger (only needed for debug information)
    bool firstEvent = true;                 // Flag for the first event processed

    TStopwatch timer;
    timer.Start();
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (size_t collID = 0; collID < irecords.size(); ++collID) {
      LOGF(debug, "Collision %lu out of %lu at %.1f ns started processing. Current pileup container size: %lu. Current number of digits accumulated: %lu",
           collID, irecords.size(), irecords[collID].getTimeNS(), mDigitizer.getPileupSignals().size(), digitsAccum.size());
      currentTime = irecords[collID];
      // Trigger logic implemented here
      bool isNewTrigger = true; // flag newly accepted readout trigger
      if (firstEvent) {
        triggerTime = currentTime;
        firstEvent = false;
      } else {
        double dT = currentTime.getTimeNS() - triggerTime.getTimeNS();
        if (dT < mParams.busyTimeNS()) {
          // busyTimeNS = readoutTimeNS + deadTimeNS, if less than that, pile up the signals and update the last time
          LOGF(debug, "Collision %lu Not creating new trigger at time %.2f since dT=%.2f ns < busy time of %.1f us", collID, currentTime.getTimeNS(), dT, mParams.busyTimeNS() / 1000);
          isNewTrigger = false;
          mDigitizer.pileup();
        } else {
          // A new signal can be received, and the detector read it out:
          // flush previous stored digits, labels and keep a trigger record
          // then update the trigger time to the new one
          if (mDigitizer.getPileupSignals().size() > 0) {
            // in case the pileup container is not empty only signal stored in there is considered
            mDigitizer.pileup(); // so we have to move the signals from the previous collision into the pileup container here
          }
          mDigitizer.flush(digits, labels);
          LOGF(debug, "Collision %lu we got %lu digits and %lu labels. There are %lu pileup containers remaining", currTrig, digits.size(), labels.getNElements(), mDigitizer.getPileupSignals().size());
          assert(digits.size() == labels.getIndexedSize());
          // Add trigger record, and send digits to the accumulator
          triggers.emplace_back(triggerTime, digitsAccum.size(), digits.size());
          std::copy(digits.begin(), digits.end(), std::back_inserter(digitsAccum));
          if (mctruth) {
            labelsAccum.mergeAtBack(labels);
          }
          triggerTime = currentTime;
          digits.clear();
          labels.clear();
          if (triggerTime.getTimeNS() - previousTime.getTimeNS() > mParams.busyTimeNS()) {
            // we safely clear all pileup signals, because any previous collision cannot contribute signal anymore
            mDigitizer.clearPileupSignals();
          }
        }
      }

      mDigitizer.setEventTime(currentTime.getTimeNS());
      if (isNewTrigger) {
        mDigitizer.setTriggerTime(triggerTime.getTimeNS());
        currTrig = collID;
      }

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        mDigitizer.setEventID(part.entryID);
        mDigitizer.setSrcID(part.sourceID);
        // get the hits for this event and this source and process them
        std::vector<o2::trd::Hit> hits;
        context->retrieveHits(mSimChains, "TRDHit", part.sourceID, part.entryID, &hits);
        LOGF(debug, "Collision %lu processing in total %lu hits", collID, hits.size());
        mDigitizer.process(hits);
      }
      previousTime = currentTime;
    }

    // Force flush of the digits that remain in the digitizer cache
    if (mDigitizer.getPileupSignals().size() > 0) {
      // remember to move signals to pileup container in case it is not empty
      mDigitizer.pileup();
    }
    mDigitizer.flush(digits, labels);
    LOGF(debug, "Collision %lu we got %lu digits and %lu labels. There are %lu pileup containers remaining", currTrig, digits.size(), labels.getNElements(), mDigitizer.getPileupSignals().size());
    assert(digits.size() == labels.getIndexedSize());
    triggers.emplace_back(triggerTime, digitsAccum.size(), digits.size());
    std::copy(digits.begin(), digits.end(), std::back_inserter(digitsAccum));
    if (mctruth) {
      labelsAccum.mergeAtBack(labels);
    }
    LOGF(info, "List of TRD chambers with at least one drift velocity out of range: %s", mDigitizer.dumpFlaggedChambers());
    timer.Stop();
    LOGF(info, "TRD digitization timing: Cpu: %.3e Real: %.3e s", timer.CpuTime(), timer.RealTime());

    LOG(info) << "TRD: Sending " << digitsAccum.size() << " digits";
    pc.outputs().snapshot(Output{"TRD", "DIGITS", 1}, digitsAccum);
    if (mctruth) {
      LOG(info) << "TRD: Sending " << labelsAccum.getNElements() << " labels";
      // we are flattening the labels and write to managed shared memory container for further communication
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{"TRD", "LABELS", 0});
      labelsAccum.flatten_to(sharedlabels);
    }
    LOG(info) << "TRD: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"TRD", "ROMode", 0}, mROMode);
    LOG(info) << "TRD: Sending trigger records";
    pc.outputs().snapshot(Output{"TRD", "TRKTRGRD", 1}, triggers);
    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    finished = true;
  }

 private:
  Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  const TRDSimParams& mParams{TRDSimParams::Instance()};
  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
};                                                                                // namespace trd

o2::framework::DataProcessorSpec getTRDDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "DIGITS", 1, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRKTRGRD", 1, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("TRD", "LABELS", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("TRD", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "TRDDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    outputs,

    AlgorithmSpec{adaptFromTask<TRDDPLDigitizerTask>()},
    Options{}};
}

} // namespace trd
} // end namespace o2
