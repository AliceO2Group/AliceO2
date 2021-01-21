// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FDDDigitizerSpec.h"
#include "TChain.h"
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
    LOG(INFO) << "initializing FDD digitization";

    //mDigitizer.setCCDBServer(dopt.ccdb);
    mDigitizer.init();
    //mROMode = mDigitizer.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
  }
  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    LOG(INFO) << "Doing FDD digitization";

    // TODO: this should eventually come from the framework and depend on the TF timestamp
    //mDigitizer.refreshCCDB();

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();

    context->initSimChains(o2::detectors::DetID::FDD, mSimChains);
    mDigitizer.setEventTime(context->getGRP().getTimeStart());
    for (auto& record : irecords) {
      LOG(INFO) << "FDD TIME RECEIVED " << record.getTimeNS();
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
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found FDD " << hits.size() << " hits ";

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
    pc.outputs().snapshot(Output{"FDD", "DIGITSBC", 0, Lifetime::Timeframe}, mDigitsBC);
    pc.outputs().snapshot(Output{"FDD", "DIGITSCH", 0, Lifetime::Timeframe}, mDigitsCh);
    pc.outputs().snapshot(Output{"FDD", "TRIGGERINPUT", 0, Lifetime::Timeframe}, mDigitsTrig);
    if (pc.outputs().isAllowed({"FDD", "DIGITLBL", 0})) {
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::fdd::MCLabel>>(Output{"FDD", "DIGITLBL", 0, Lifetime::Timeframe});
      labels.flatten_to(sharedlabels);
      labels.clear_andfreememory();
    }

    LOG(INFO) << "FDD: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"FDD", "ROMode", 0, Lifetime::Timeframe}, mROMode);

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
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getFDDDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
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
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<FDDDPLDigitizerTask>()},
    Options{}};
}
} // namespace fdd
} // namespace o2
