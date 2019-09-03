// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"
#include <SimulationDataFormat/MCTruthContainer.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ZDCSimulation/Digit.h"
#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/Detector.h"
#include "ZDCSimulation/MCLabel.h"
#include "DetectorsBase/GeometryManager.h"

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
namespace zdc
{

class ZDCDPLDigitizerTask
{
 public:
  void init(framework::InitContext& ic)
  {
    LOG(INFO) << "Initializing ZDC digitization";
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
  }

  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    LOG(INFO) << "Doing ZDC digitization";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();

    for (auto& record : irecords) {
      LOG(INFO) << "ZDC TIME RECEIVED " << record.timeNS;
    }

    auto& eventParts = context->getEventParts();

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    std::vector<o2::zdc::Hit> hits;

    for (int collID = 0; collID < irecords.size(); ++collID) {

      const auto& irec = irecords[collID];
      mDigitizer.setInteractionRecord(irec);

      for (auto& part : eventParts[collID]) {

        retrieveHits(mSimChains, "ZDCHit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found ZDC " << hits.size() << " hits ";

        mDigitizer.setEventID(part.entryID);
        mDigitizer.setSrcID(part.sourceID);

        mDigitizer.process(hits, mDigits, mLabels);
      }
    }

    o2::InteractionTimeRecord terminateIR;
    terminateIR.orbit = 0xffffffff; // supply IR in the infinite future to flush all cached BC
    mDigitizer.setInteractionRecord(terminateIR);
    mDigitizer.flush(mDigits, mLabels);

    // send out to next stage
    pc.outputs().snapshot(Output{"ZDC", "DIGITS", 0, Lifetime::Timeframe}, mDigits);
    pc.outputs().snapshot(Output{"ZDC", "DIGITLBL", 0, Lifetime::Timeframe}, mLabels);

    LOG(INFO) << "ZDC: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"ZDC", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    mFinished = true;
  }

 private:
  bool mFinished = false;
  Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  std::vector<o2::zdc::Digit> mDigits;
  o2::dataformats::MCTruthContainer<o2::zdc::MCLabel> mLabels; // labels which get filled

  // RS: at the moment using hardcoded flag for continuous readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getZDCDigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "ZDCDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    Outputs{OutputSpec{"ZDC", "DIGITS", 0, Lifetime::Timeframe},
            OutputSpec{"ZDC", "DIGITLBL", 0, Lifetime::Timeframe},
            OutputSpec{"ZDC", "ROMode", 0, Lifetime::Timeframe}},

    AlgorithmSpec{adaptFromTask<ZDCDPLDigitizerTask>()},

    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}}}};
}

} // end namespace zdc
} // end namespace o2
