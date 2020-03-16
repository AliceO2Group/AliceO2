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
#include "Steer/HitProcessingManager.h" // for RunContext
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCTruthContainer.h"
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
namespace fdd
{

class FDDDPLDigitizerTask
{
  using GRP = o2::parameters::GRPObject;

 public:
  void init(framework::InitContext& ic)
  {
    LOG(INFO) << "Initializing FDD digitization";

    //auto& dopt = o2::conf::DigiParams::Instance();

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
    mDigitizer.setEventTime(grp->getTimeStart());
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
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();
    auto& eventParts = context->getEventParts();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    std::vector<o2::fdd::Hit> hits;

    for (int collID = 0; collID < irecords.size(); ++collID) {

      const auto& irec = irecords[collID];
      mDigitizer.setInteractionRecord(irec);

      for (auto& part : eventParts[collID]) {

        retrieveHits(mSimChains, "FDDHit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found FDD " << hits.size() << " hits ";

        mDigitizer.setEventID(part.entryID);
        mDigitizer.setSrcID(part.sourceID);

        mDigitizer.process(hits, mDigitsBC, mDigitsCh, mLabels);
      }
    }

    o2::InteractionTimeRecord terminateIR;
    terminateIR.orbit = 0xffffffff; // supply IR in the infinite future to flush all cached BC
    mDigitizer.setInteractionRecord(terminateIR);
    mDigitizer.flush(mDigitsBC, mDigitsCh, mLabels);

    // send out to next stage
    pc.outputs().snapshot(Output{"FDD", "DIGITSBC", 0, Lifetime::Timeframe}, mDigitsBC);
    pc.outputs().snapshot(Output{"FDD", "DIGITSCH", 0, Lifetime::Timeframe}, mDigitsCh);
    pc.outputs().snapshot(Output{"FDD", "DIGITLBL", 0, Lifetime::Timeframe}, mLabels);

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
  o2::dataformats::MCTruthContainer<o2::fdd::MCLabel> mLabels; // labels which get filled

  // RS: at the moment using hardcoded flag for continuous readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getFDDDigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "FDDDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    Outputs{OutputSpec{"FDD", "DIGITSBC", 0, Lifetime::Timeframe},
            OutputSpec{"FDD", "DIGITSCH", 0, Lifetime::Timeframe},
            OutputSpec{"FDD", "DIGITLBL", 0, Lifetime::Timeframe},
            OutputSpec{"FDD", "ROMode", 0, Lifetime::Timeframe}},

    AlgorithmSpec{adaptFromTask<FDDDPLDigitizerTask>()},

    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}}}};
}
} // namespace fdd
} // namespace o2
