// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"
#include "FITSimulation/Digitizer.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "Framework/Task.h"
#include <iostream>
#include "DataFormatsParameters/GRPObject.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace fit
{

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

class FITDPLDigitizerTask
{
 public:
  FITDPLDigitizerTask() = default;
  ~FITDPLDigitizerTask() = default;

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

    mDigitizer.init();
    const bool isContinuous = ic.options().get<int>("pileup");
    // mDigitizer.setContinuous(isContinuous);
    // mDigitizer.setMCTruthContainer(labels.get());
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
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

    LOG(INFO) << "CALLING FIT DIGITIZATION";

    static std::vector<o2::fit::HitType> hits;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;
    o2::fit::Digit digit;
    std::vector<o2::fit::Digit> digitAccum; // digit accumulator

    auto& eventParts = context->getEventParts();
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      mDigitizer.setEventTime(timesview[collID].timeNS);

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        mDigitizer.setEventID(part.entryID);
        // mDigitizer.setSrcID(part.sourceID);

        // get the hits for this event and this source
        hits.clear();
        retrieveHits(mSimChains, "FITHit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found " << hits.size() << " hits ";

        // call actual digitization procedure
        labels.clear();
        // digits.clear();
        mDigitizer.process(&hits, &digit);
        digit.printStream(std::cout);
        auto data = digit.getChDgData();
        LOG(INFO) << "Have " << data.size() << " fired channels ";
        // copy digits into accumulator
        digitAccum.push_back(digit); // we should move it there actually
        LOG(INFO) << "Have " << digitAccum.back().getChDgData().size() << " fired channels ";
        // labelAccum.mergeAtBack(*labels);
        // LOG(INFO) << "Have " << digits->size() << " digits ";
      }
    }
    //    if (mDigitizer.isContinuous()) {
    //      digits->clear();
    //      labels->clear();
    //      digitizer->flushOutputContainer(*digits.get());
    //      LOG(INFO) << "FLUSHING LEFTOVER STUFF " << digits->size();
    //      // copy digits into accumulator
    //      std::copy(digits->begin(), digits->end(), std::back_inserter(*digitsAccum.get()));
    //      labelAccum.mergeAtBack(*labels);
    //    }

    // LOG(INFO) << "Have " << labelAccum.getNElements() << " TOF labels ";
    // here we have all digits and we can send them to consumer (aka snapshot it onto output)
    pc.outputs().snapshot(Output{ "FIT", "DIGITS", 0, Lifetime::Timeframe }, digitAccum);
    // pc.outputs().snapshot(Output{ "FIT", "DIGITSMCTR", 0, Lifetime::Timeframe }, labelAccum);
    LOG(INFO) << "FIT: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{ "FIT", "ROMode", 0, Lifetime::Timeframe }, mROMode);
    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  }

 private:
  Bool_t mContinuous = kFALSE;  ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns

  Digitizer mDigitizer; ///< Digitizer
  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode

  std::vector<TChain*> mSimChains;
};

o2::framework::DataProcessorSpec getFITDigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "FITDigitizer", Inputs{ InputSpec{ "collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe } },
    Outputs{ OutputSpec{ "FIT", "DIGITS", 0, Lifetime::Timeframe },
             /*OutputSpec{ "FIT", "DIGITSMCTR", 0, Lifetime::Timeframe }*/
             OutputSpec{ "FIT", "ROMode", 0, Lifetime::Timeframe } },
    AlgorithmSpec{ adaptFromTask<FITDPLDigitizerTask>() },
    Options{ { "simFile", VariantType::String, "o2sim.root", { "Sim (background) input filename" } },
             { "simFileS", VariantType::String, "", { "Sim (signal) input filename" } },
             { "pileup", VariantType::Int, 1, { "whether to run in continuous time mode" } } }
    // I can't use VariantType::Bool as it seems to have a problem
  };
}

} // end namespace fit
} // end namespace o2
