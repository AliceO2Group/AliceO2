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

#include "FDDDigitizerSpec.h"
#include "TChain.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "FDDSimulation/Digitizer.h"
#include "FDDBase/Geometry.h"
#include "FDDSimulation/DigitizationParameters.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DataFormatsFDD/MCLabel.h"
#include "DataFormatsFIT/DeadChannelMap.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace fdd
{

class FDDDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
  using GRP = o2::parameters::GRPObject;

 public:
  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(info) << "initializing FDD digitization";

    //mDigitizer.setCCDBServer(dopt.ccdb);
    mDigitizer.init();
    //mROMode = mDigitizer.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
    mUseDeadChannelMap = !ic.options().get<bool>("disable-dead-channel-map");
    mUpdateDeadChannelMap = mUseDeadChannelMap;
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    // Initialize the dead channel map only once
    if (matcher == ConcreteDataMatcher("FDD", "DeadChannelMap", 0)) {
      mUpdateDeadChannelMap = false;
    }
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    LOG(info) << "Doing FDD digitization";

    // TODO: this should eventually come from the framework and depend on the TF timestamp
    //mDigitizer.refreshCCDB();

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();

    context->initSimChains(o2::detectors::DetID::FDD, mSimChains);
    mDigitizer.setEventTime(context->getGRP().getTimeStart());
    for (auto& record : irecords) {
      LOG(info) << "FDD TIME RECEIVED " << record.getTimeNS();
    }

    // Initialize the dead channel map
    if (mUpdateDeadChannelMap && mUseDeadChannelMap) {
      auto deadChannelMap = pc.inputs().get<o2::fit::DeadChannelMap*>("fdddeadchannelmap");
      mDigitizer.setDeadChannelMap(deadChannelMap.get());
    }

    auto& eventParts = context->getEventParts();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    std::vector<o2::fdd::Hit> hits;
    o2::dataformats::MCTruthContainer<o2::fdd::MCLabel> labels;

    for (int collID = 0; collID < irecords.size(); ++collID) {

      const auto& irec = irecords[collID];
      mDigitizer.setInteractionRecord(irec);

      for (auto& part : eventParts[collID]) {

        // get the hits for this event and this source
        context->retrieveHits(mSimChains, "FDDHit", part.sourceID, part.entryID, &hits);
        LOG(info) << "For collision " << collID << " eventID " << part.entryID << " found FDD " << hits.size() << " hits ";

        mDigitizer.setEventID(part.entryID);
        mDigitizer.setSrcID(part.sourceID);

        mDigitizer.process(hits, mDigitsBC, mDigitsCh, mDigitsTrig, labels);
      }
    }

    o2::InteractionTimeRecord terminateIR;
    terminateIR.orbit = 0xffffffff; // supply IR in the infinite future to flush all cached BC
    mDigitizer.setInteractionRecord(terminateIR);
    mDigitizer.flush(mDigitsBC, mDigitsCh, mDigitsTrig, labels);

    // send out to next stage
    pc.outputs().snapshot(Output{"FDD", "DIGITSBC", 0}, mDigitsBC);
    pc.outputs().snapshot(Output{"FDD", "DIGITSCH", 0}, mDigitsCh);
    pc.outputs().snapshot(Output{"FDD", "TRIGGERINPUT", 0}, mDigitsTrig);
    if (pc.outputs().isAllowed({"FDD", "DIGITLBL", 0})) {
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::fdd::MCLabel>>(Output{"FDD", "DIGITLBL", 0});
      labels.flatten_to(sharedlabels);
      labels.clear_andfreememory();
    }

    LOG(info) << "FDD: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"FDD", "ROMode", 0}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 private:
  bool mFinished = false;
  Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  std::vector<o2::fdd::ChannelData> mDigitsCh;
  std::vector<o2::fdd::Digit> mDigitsBC;
  std::vector<o2::fdd::DetTrigInput> mDigitsTrig;

  // RS: at the moment using hardcoded flag for continuous readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::ROMode(o2::parameters::GRPObject::CONTINUOUS | o2::parameters::GRPObject::TRIGGERING); // readout mode

  bool mUseDeadChannelMap = true;
  bool mUpdateDeadChannelMap = true;
};

o2::framework::DataProcessorSpec getFDDDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<InputSpec> inputs;
  inputs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputs.emplace_back("fdddeadchannelmap", "FDD", "DeadChannelMap", 0,
                      Lifetime::Condition,
                      ccdbParamSpec("FDD/Calib/DeadChannelMap", {}, -1));
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("FDD", "DIGITSBC", 0, Lifetime::Timeframe);
  outputs.emplace_back("FDD", "DIGITSCH", 0, Lifetime::Timeframe);
  outputs.emplace_back("FDD", "TRIGGERINPUT", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("FDD", "DIGITLBL", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("FDD", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "FDDDigitizer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<FDDDPLDigitizerTask>()},
    Options{{"disable-dead-channel-map", VariantType::Bool, false, {"Don't mask dead channels"}}}};
}
} // namespace fdd
} // namespace o2
