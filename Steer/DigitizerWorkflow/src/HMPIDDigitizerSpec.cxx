// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDDigitizerSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDSimulation/HMPIDDigitizer.h"
#include "HMPIDSimulation/Detector.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <SimConfig/DigiParams.h>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;


namespace o2
{
namespace hmpid
{

class HMPIDDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
 public:
  HMPIDDPLDigitizerTask() : o2::base::BaseDPLDigitizer(o2::base::InitServices::GEOM) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    LOG(INFO) << "Doing HMPID digitization";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");

    context->initSimChains(o2::detectors::DetID::HMP, mSimChains);

    auto& irecords = context->getEventRecords();
    for (auto& record : irecords) {
      LOG(INFO) << "HMPID TIME RECEIVED " << record.getTimeNS();
    }

    auto& eventParts = context->getEventParts();
    std::vector<o2::hmpid::raw::Digit> digitsAccum;                // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum; // timeframe accumulator for labels
    mIntRecord.clear();

    auto flushDigitsAndLabels = [this, &digitsAccum, &labelAccum]() {
      // flush previous buffer
      mDigits.clear();
      mLabels.clear();
      mDigitizer.flush(mDigits);
      LOG(INFO) << "HMPID flushed " << mDigits.size() << " digits at this time ";
      LOG(INFO) << "NUMBER OF LABEL OBTAINED " << mLabels.getNElements();
      int32_t first = digitsAccum.size(); // this is the first
      std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(digitsAccum));
      int32_t last = digitsAccum.size() - 1; // this is the last
      labelAccum.mergeAtBack(mLabels);

      // save info for the triggers accepted
      LOG(INFO) << "Trigger  Orbit :" << mDigitizer.getOrbit() << "  BC:" << mDigitizer.getBc();
      mIntRecord.push_back(o2::hmpid::raw::Event(o2::InteractionRecord(mDigitizer.getBc(), mDigitizer.getOrbit()), first, last));
    };

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < irecords.size(); ++collID) {
      // try to start new readout cycle by setting the trigger time
      auto triggeraccepted = mDigitizer.setTriggerTime(irecords[collID].getTimeNS());
      if (triggeraccepted) {
        flushDigitsAndLabels(); // flush previous readout cycle
      }
      auto withinactivetime = mDigitizer.setEventTime(irecords[collID].getTimeNS());
      if (withinactivetime) {
        // for each collision, loop over the constituents event and source IDs
        // (background signal merging is basically taking place here)
        for (auto& part : eventParts[collID]) {
          mDigitizer.setEventID(part.entryID);
          mDigitizer.setSrcID(part.sourceID);

          // get the hits for this event and this source
          std::vector<o2::hmpid::raw::HitType> hits;
          context->retrieveHits(mSimChains, "HMPHit", part.sourceID, part.entryID, &hits);
          LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found HMP " << hits.size() << " hits ";

          mDigitizer.setLabelContainer(&mLabels);
          mLabels.clear();
          mDigits.clear();

          mDigitizer.process(hits, mDigits);
        }

      } else {
        LOG(INFO) << "COLLISION " << collID << "FALLS WITHIN A DEAD TIME";
      }
    }
    // final flushing step; getting everything not yet written out
    flushDigitsAndLabels();

    // send out to next stage
    pc.outputs().snapshot(Output{"HMP", "DIGITS", 0, Lifetime::Timeframe}, digitsAccum);
    pc.outputs().snapshot(Output{"HMP", "INTRECORDS", 0, Lifetime::Timeframe}, mIntRecord);
    if (pc.outputs().isAllowed({"HMP", "DIGITLBL", 0})) {
      pc.outputs().snapshot(Output{"HMP", "DIGITLBL", 0, Lifetime::Timeframe}, labelAccum);
    }
    LOG(INFO) << "HMP: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"HMP", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    finished = true;
  }

 private:
  HMPIDDigitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  std::vector<o2::hmpid::raw::Digit> mDigits;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels; // labels which get filled
  std::vector<o2::hmpid::raw::Event> mIntRecord;

  // RS: at the moment using hardcoded flag for continuous readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getHMPIDDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("HMP", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("HMP", "INTRECORDS", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("HMP", "DIGITLBL", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("HMP", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "HMPIDDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    outputs,
    AlgorithmSpec{adaptFromTask<HMPIDDPLDigitizerTask>()}, Options{}};
}

} // end namespace hmpid
} // end namespace o2
