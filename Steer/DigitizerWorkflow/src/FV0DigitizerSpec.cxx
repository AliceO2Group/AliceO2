// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FV0DigitizerSpec.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/BCData.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include <TStopwatch.h>
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include <TChain.h>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "FV0Simulation/Digitizer.h"
#include "FV0Simulation/DigitizationConstant.h"
#include "FV0Simulation/MCLabel.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include <TFile.h>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace fv0
{

class FV0DPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
  using GRP = o2::parameters::GRPObject;

 public:
  FV0DPLDigitizerTask() : o2::base::BaseDPLDigitizer(), mDigitizer(), mSimChains(), mDigitsCh(), mDigitsBC(), mLabels() {}
  ~FV0DPLDigitizerTask() override = default;

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(INFO) << "FV0DPLDigitizerTask:init";
    mDigitizer.init();
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    LOG(INFO) << "FV0DPLDigitizerTask:run";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(o2::detectors::DetID::FV0, mSimChains);

    mDigitizer.setTimeStamp(context->getGRP().getTimeStart());

    auto& irecords = context->getEventRecords();
    auto& eventParts = context->getEventParts();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    std::vector<o2::fv0::Hit> hits;
    for (int collID = 0; collID < irecords.size(); ++collID) {
      mDigitizer.clear();
      const auto& irec = irecords[collID];
      mDigitizer.setInteractionRecord(irec);
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        hits.clear();
        context->retrieveHits(mSimChains, "FV0Hit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "[FV0] For collision " << collID << " eventID " << part.entryID << " found " << hits.size() << " hits ";

        // call actual digitization procedure
        mDigitizer.setEventId(part.entryID);
        mDigitizer.setSrcId(part.sourceID);
        mDigitizer.process(hits);
      }
      mDigitizer.analyseWaveformsAndStore(mDigitsBC, mDigitsCh, mLabels);
      LOG(INFO) << "[FV0] Has " << mDigitsBC.size() << " BC elements,   " << mDigitsCh.size() << " mDigitsCh elements";
    }

    // here we have all digits and we can send them to consumer (aka snapshot it onto output)
    LOG(INFO) << "FV0: Sending " << mDigitsBC.size() << " digitsBC and " << mDigitsCh.size() << " digitsCh.";

    // send out to next stage
    pc.outputs().snapshot(Output{"FV0", "DIGITSBC", 0, Lifetime::Timeframe}, mDigitsBC);
    pc.outputs().snapshot(Output{"FV0", "DIGITSCH", 0, Lifetime::Timeframe}, mDigitsCh);
    pc.outputs().snapshot(Output{"FV0", "DIGITLBL", 0, Lifetime::Timeframe}, mLabels);

    LOG(INFO) << "FV0: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"FV0", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 private:
  bool mFinished = false;
  Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  std::vector<o2::fv0::ChannelData> mDigitsCh;
  std::vector<o2::fv0::BCData> mDigitsBC;
  o2::dataformats::MCTruthContainer<o2::fv0::MCLabel> mLabels; // labels which get filled

  // RS: at the moment using hardcoded flag for continuous readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getFV0DigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "FV0Digitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    Outputs{OutputSpec{"FV0", "DIGITSBC", 0, Lifetime::Timeframe},
            OutputSpec{"FV0", "DIGITSCH", 0, Lifetime::Timeframe},
            OutputSpec{"FV0", "DIGITLBL", 0, Lifetime::Timeframe},
            OutputSpec{"FV0", "ROMode", 0, Lifetime::Timeframe}},

    AlgorithmSpec{adaptFromTask<FV0DPLDigitizerTask>()},

    Options{{"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}};
}

} // end namespace fv0
} // end namespace o2
