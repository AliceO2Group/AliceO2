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
#include "Steer/HitProcessingManager.h" // for RunContext
#include <TChain.h>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "FV0Simulation/Digitizer.h"
#include "FV0Simulation/DigitizationParameters.h"
#include "FV0Simulation/MCLabel.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TFile.h>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

// helper function which will be offered as a service
template <typename T>
void retrieveHits(std::vector<TChain*> const& chains,
                  const char* brname,
                  int sourceID,
                  int entryID,
                  std::vector<T>* hits)
{
  auto br = chains[sourceID]->GetBranch(brname);
  if (!br) {
    LOG(ERROR) << "No branch found";
    return;
  }
  br->SetAddress(&hits);
  br->GetEntry(entryID);
}

namespace o2
{
namespace fv0
{

class FV0DPLDigitizerTask
{
  using GRP = o2::parameters::GRPObject;

 public:
  FV0DPLDigitizerTask()
    : mDigitizer(), mSimChains(), mDigitsCh(), mDigitsBC(), mLabels() {}
  ~FV0DPLDigitizerTask() = default;

  void init(framework::InitContext& ic)
  {
    LOG(INFO) << "FV0DPLDigitizerTask:init";

    // setup the input chain for the hits
    mSimChains.emplace_back(new TChain("o2sim"));

    // add the main (background) file
    mSimChains.back()->AddFile(ic.options().get<std::string>("simFile").c_str());

    // maybe add a particular signal file
    auto signalfilename = ic.options().get<std::string>("simFileS");
    if (signalfilename.size() > 0) {
      mSimChains.emplace_back(new TChain("o2sim"));
      mSimChains.back()->AddFile(signalfilename.c_str());
    }

    const std::string inputGRP = "o2sim_grp.root";
    const std::string grpName = "GRP";
    TFile flGRP(inputGRP.c_str());
    if (flGRP.IsZombie()) {
      LOG(FATAL) << "Failed to open " << inputGRP;
    }
    std::unique_ptr<GRP> grp(static_cast<GRP*>(flGRP.GetObjectChecked(grpName.c_str(), GRP::Class())));
    mDigitizer.setTimeStamp(grp->getTimeStart());
    mDigitizer.init();
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    LOG(INFO) << "FV0DPLDigitizerTask:run";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();
    auto& eventParts = context->getEventParts();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    std::vector<o2::fv0::Hit> hits;
    for (int collID = 0; collID < irecords.size(); ++collID) {
      const auto& irec = irecords[collID];
      mDigitizer.setInteractionRecord(irec);
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        mDigitizer.clear();
        hits.clear();

        retrieveHits(mSimChains, "FV0Hit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "[FV0] For collision " << collID << " eventID " << part.entryID << " found " << hits.size() << " hits ";

        // call actual digitization procedure
        //        labels.clear();
        mDigitizer.setEventId(part.entryID);
        mDigitizer.setSrcId(part.sourceID);
        mDigitizer.process(hits, mDigitsBC, mDigitsCh, mLabels);
        LOG(INFO) << "[FV0] Has " << mDigitsBC.size() << " BC elements,   " << mDigitsCh.size() << " mDigitsCh elements";
      }
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

    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
            {"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}

    // I can't use VariantType::Bool as it seems to have a problem
  };
}

} // end namespace fv0
} // end namespace o2
