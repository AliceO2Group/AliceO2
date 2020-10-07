// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "TRDBase/Digit.h" // for the Digit type
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/Detector.h" // for the Hit type
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "TRDBase/Calibrations.h"
#include "DataFormatsTRD/TriggerRecord.h"

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
    LOG(INFO) << "initializing TRD digitization";
    mDigitizer.init();
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    LOG(INFO) << "Doing TRD digitization";

    bool mctruth = pc.outputs().isAllowed({"TRD", "LABELS", 0});

    Calibrations simcal;
    simcal.setCCDBForSimulation(297595);
    mDigitizer.setCalibrations(&simcal);

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(o2::detectors::DetID::TRD, mSimChains);
    auto& irecords = context->getEventRecords();

    for (auto& record : irecords) {
      LOG(INFO) << "TRD TIME RECEIVED " << record.getTimeNS();
    }

    auto& eventParts = context->getEventParts();
    std::vector<o2::trd::Digit> digitsAccum; // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::trd::MCLabel> labelsAccum;
    std::vector<TriggerRecord> triggers;

    std::vector<o2::trd::Digit> digits;                         // digits which get filled
    o2::dataformats::MCTruthContainer<o2::trd::MCLabel> labels; // labels which get filled

    TStopwatch timer;
    timer.Start();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < irecords.size(); ++collID) {
      mDigitizer.setEventTime(irecords[collID].getTimeNS());

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        mDigitizer.setEventID(part.entryID);
        mDigitizer.setSrcID(part.sourceID);

        // get the hits for this event and this source
        std::vector<o2::trd::HitType> hits;
        context->retrieveHits(mSimChains, "TRDHit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found TRD " << hits.size() << " hits ";

        mDigitizer.process(hits, digits, labels);
        assert(digits.size() == labels.getIndexedSize());

        // Add trigger record
        triggers.emplace_back(irecords[collID], digitsAccum.size(), digits.size());

        std::copy(digits.begin(), digits.end(), std::back_inserter(digitsAccum));
        if (mctruth) {
          labelsAccum.mergeAtBack(labels);
        }
        digits.clear();
        labels.clear();
      }
    }
    // Force flush of the digits that remain in the digitizer cache
    mDigitizer.flush(digits, labels);
    assert(digits.size() == labels.getIndexedSize());

    triggers.emplace_back(irecords[irecords.size() - 1], digitsAccum.size(), digits.size());
    std::copy(digits.begin(), digits.end(), std::back_inserter(digitsAccum));
    if (mctruth) {
      labelsAccum.mergeAtBack(labels);
    }
    timer.Stop();
    LOG(INFO) << "TRD: Digitization took " << timer.RealTime() << "s";

    LOG(INFO) << "TRD: Sending " << digitsAccum.size() << " digits";
    pc.outputs().snapshot(Output{"TRD", "DIGITS", 0, Lifetime::Timeframe}, digitsAccum);
    if (mctruth) {
      LOG(INFO) << "TRD: Sending " << labelsAccum.getNElements() << " labels";
      // we are flattening the labels and write to managed shared memory container for further communication
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::trd::MCLabel>>(Output{"TRD", "LABELS", 0, Lifetime::Timeframe});
      labelsAccum.flatten_to(sharedlabels);
    }
    LOG(INFO) << "TRD: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"TRD", "ROMode", 0, Lifetime::Timeframe}, mROMode);
    LOG(INFO) << "TRD: Sending trigger records";
    pc.outputs().snapshot(Output{"TRD", "TRGRDIG", 0, Lifetime::Timeframe}, triggers);
    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    finished = true;
  }

 private:
  Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT; // readout mode
};

o2::framework::DataProcessorSpec getTRDDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRGRDIG", 0, Lifetime::Timeframe);
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

} // end namespace trd
} // end namespace o2
