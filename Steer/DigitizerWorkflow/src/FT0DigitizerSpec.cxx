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
#include "Steer/HitProcessingManager.h" // for RunContext
#include "FT0Simulation/Digitizer.h"
#include "FT0Simulation/DigitizationParameters.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/HitType.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/MCLabel.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include <TChain.h>
#include <iostream>
#include <TStopwatch.h>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace ft0
{
// helper function which will be offered as a service
//template <typename T>

class FT0DPLDigitizerTask
{

  using GRP = o2::parameters::GRPObject;

 public:
  FT0DPLDigitizerTask() : mDigitizer(DigitizationParameters{}) {}
  explicit FT0DPLDigitizerTask(o2::ft0::DigitizationParameters const& parameters)
    : mDigitizer(parameters){};
  ~FT0DPLDigitizerTask() = default;

  void init(framework::InitContext& ic)
  {
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

    mDigitizer.init();
    mROMode = mDigitizer.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
  }

  void run(framework::ProcessingContext& pc)
  {

    if (mFinished) {
      return;
    }

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& timesview = context->getEventRecords();

    // if there is nothing to do ... return
    if (timesview.size() == 0) {
      return;
    }

    TStopwatch timer;
    timer.Start();

    LOG(INFO) << "CALLING FT0 DIGITIZATION";

    static std::vector<o2::ft0::HitType> hits;
    o2::dataformats::MCTruthContainer<o2::ft0::MCLabel> labelAccum;
    o2::dataformats::MCTruthContainer<o2::ft0::MCLabel> labels;

    mDigitizer.setMCLabels(&labels);
    auto& eventParts = context->getEventParts();
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      mDigitizer.setTimeStamp(timesview[collID].timeNS);
      mDigitizer.setInteractionRecord(timesview[collID]);
      mDigitizer.clearDigits();
      std::vector<std::vector<double>> channel_times;
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        // get the hits for this event and this source
        hits.clear();
        retrieveHits(mSimChains, part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " source ID " << part.sourceID << " found " << hits.size() << " hits ";

        // call actual digitization procedure
        labels.clear();
        mDigitizer.setEventID(collID);
        mDigitizer.setSrcID(part.sourceID);
        mDigitizer.process(&hits);
        // copy labels into accumulator
        labelAccum.mergeAtBack(labels);
      }
      mDigitizer.setDigits(mDigitsBC, mDigitsCh);
    }

    // send out to next stage
    pc.outputs().snapshot(Output{"FT0", "DIGITSBC", 0, Lifetime::Timeframe}, mDigitsBC);
    pc.outputs().snapshot(Output{"FT0", "DIGITSCH", 0, Lifetime::Timeframe}, mDigitsCh);
    pc.outputs().snapshot(Output{"FT0", "DIGITSMCTR", 0, Lifetime::Timeframe}, labelAccum);

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

  Bool_t mContinuous = kFALSE;   ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1;  ///< Fair time unit in ns
  o2::ft0::Digitizer mDigitizer; ///< Digitizer

  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode

  std::vector<TChain*> mSimChains;

  void retrieveHits(std::vector<TChain*> const& chains,
                    int sourceID,
                    int entryID,
                    std::vector<o2::ft0::HitType>* hits)
  {
    auto br = mSimChains[sourceID]->GetBranch("FT0Hit");
    if (!br) {
      LOG(ERROR) << "No branch found";
      return;
    }
    br->SetAddress(&hits);
    br->GetEntry(entryID);
  }
};

o2::framework::DataProcessorSpec getFT0DigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)

  return DataProcessorSpec{
    "FT0Digitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    Outputs{OutputSpec{"FT0", "DIGITSBC", 0, Lifetime::Timeframe},
            OutputSpec{"FT0", "DIGITSCH", 0, Lifetime::Timeframe},
            OutputSpec{"FT0", "DIGITSMCTR", 0, Lifetime::Timeframe},
            OutputSpec{"FT0", "ROMode", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<FT0DPLDigitizerTask>()},
    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
            {"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}

    // I can't use VariantType::Bool as it seems to have a problem
  };
}

} // namespace ft0
} // end namespace o2
