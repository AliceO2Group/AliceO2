// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0DigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "FT0Simulation/Digitizer.h"
#include "FT0Simulation/DigitizationParameters.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/HitType.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/MCLabel.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/Task.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DataFormatsParameters/GRPObject.h"
#include <TChain.h>
#include <TStopwatch.h>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace ft0
{

class FT0DPLDigitizerTask : public o2::base::BaseDPLDigitizer
{

  using GRP = o2::parameters::GRPObject;

 public:
  FT0DPLDigitizerTask() : o2::base::BaseDPLDigitizer(), mDigitizer(DigitizationParameters{}) {}
  explicit FT0DPLDigitizerTask(o2::ft0::DigitizationParameters const& parameters)
    : o2::base::BaseDPLDigitizer(), mDigitizer(parameters){};
  ~FT0DPLDigitizerTask() override = default;

  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDigitizer.init();
    mROMode = mDigitizer.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
    mDisableQED = ic.options().get<bool>("disable-qed");
  }

  void run(framework::ProcessingContext& pc)
  {

    if (mFinished) {
      return;
    }

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(o2::detectors::DetID::FT0, mSimChains);
    const bool withQED = context->isQEDProvided() && !mDisableQED;
    auto& timesview = context->getEventRecords(withQED);

    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }

    TStopwatch timer;
    timer.Start();

    LOG(INFO) << "CALLING FT0 DIGITIZATION";

    static std::vector<o2::ft0::HitType> hits;
    // o2::dataformats::MCTruthContainer<o2::ft0::MCLabel> labelAccum;
    o2::dataformats::MCTruthContainer<o2::ft0::MCLabel> labels;

    // mDigitizer.setMCLabels(&labels);
    auto& eventParts = context->getEventParts(withQED);
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      mDigitizer.setInteractionRecord(timesview[collID]);
      LOG(DEBUG) << " setInteractionRecord " << timesview[collID] << " bc " << mDigitizer.getBC() << " orbit " << mDigitizer.getOrbit();
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        // get the hits for this event and this source
        hits.clear();
        context->retrieveHits(mSimChains, "FT0Hit", part.sourceID, part.entryID, &hits);
        LOG(DEBUG) << "For collision " << collID << " eventID " << part.entryID << " source ID " << part.sourceID << " found " << hits.size() << " hits ";
        if (hits.size() > 0) {
          // call actual digitization procedure
          mDigitizer.setEventID(part.entryID);
          mDigitizer.setSrcID(part.sourceID);
          mDigitizer.process(&hits, mDigitsBC, mDigitsCh, mDigitsTrig, labels);
        }
      }
    }
    mDigitizer.flush_all(mDigitsBC, mDigitsCh, mDigitsTrig, labels);

    // send out to next stage
    pc.outputs().snapshot(Output{"FT0", "DIGITSBC", 0, Lifetime::Timeframe}, mDigitsBC);
    pc.outputs().snapshot(Output{"FT0", "DIGITSCH", 0, Lifetime::Timeframe}, mDigitsCh);
    pc.outputs().snapshot(Output{"FT0", "TRIGGERINPUT", 0, Lifetime::Timeframe}, mDigitsTrig);
    if (pc.outputs().isAllowed({"FT0", "DIGITSMCTR", 0})) {
      pc.outputs().snapshot(Output{"FT0", "DIGITSMCTR", 0, Lifetime::Timeframe}, labels);
    }
    LOG(INFO) << "FT0: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"FT0", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 protected:
  bool mFinished = false;
  std::vector<o2::ft0::ChannelData> mDigitsCh;
  std::vector<o2::ft0::Digit> mDigitsBC;
  std::vector<o2::ft0::DetTrigInput> mDigitsTrig;

  Bool_t mContinuous = kFALSE;   ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1;  ///< Fair time unit in ns
  o2::ft0::Digitizer mDigitizer; ///< Digitizer

  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode

  //
  bool mDisableQED = false;

  std::vector<TChain*> mSimChains;
};

o2::framework::DataProcessorSpec getFT0DigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("FT0", "DIGITSBC", 0, Lifetime::Timeframe);
  outputs.emplace_back("FT0", "DIGITSCH", 0, Lifetime::Timeframe);
  outputs.emplace_back("FT0", "TRIGGERINPUT", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("FT0", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("FT0", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "FT0Digitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<FT0DPLDigitizerTask>()},
    Options{{"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}},
            {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}

} // namespace ft0
} // end namespace o2
