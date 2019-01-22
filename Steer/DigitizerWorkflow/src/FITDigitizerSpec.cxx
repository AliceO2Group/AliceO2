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
#include "T0Simulation/DigitizationParameters.h"
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

    if (mID == o2::detectors::DetID::T0)   mT0Digitizer.init();
    const bool isContinuous = ic.options().get<int>("pileup");
    // mT0Digitizer.setContinuous(isContinuous);
    // mT0Digitizer.setMCTruthContainer(labels.get());
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

    static std::vector<o2::t0::HitType> hits;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;
    o2::fit::Digit digit;
    std::vector<o2::fit::Digit> digitAccum; // digit accumulator

    auto& eventParts = context->getEventParts();
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < timesview.size(); ++collID) {
      mT0Digitizer.setEventTime(timesview[collID].timeNS);
      mT0Digitizer.setOrbit(timesview[collID].orbit);
      mT0Digitizer.setBC(timesview[collID].bc);
      digit.cleardigits();
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        // get the hits for this event and this source
        hits.clear();
        retrieveHits(mSimChains, "FITT0Hit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found " << hits.size() << " hits ";

        // call actual digitization procedure
        labels.clear();
        // digits.clear();
        mT0Digitizer.process(&hits, &digit);
        auto data = digit.getChDgData();
        LOG(INFO) << "Have " << data.size() << " fired channels ";
        // copy digits into accumulator
        // labelAccum.mergeAtBack(*labels);
      }
      mT0Digitizer.setTriggers(&digit);
      mT0Digitizer.smearCFDtime(&digit);
      digitAccum.push_back(digit); // we should move it there actually
      LOG(INFO) << "Have " << digitAccum.back().getChDgData().size() << " fired channels ";
      digit.printStream(std::cout);
    }
    //    if (mT0Digitizer.isContinuous()) {
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
    pc.outputs().snapshot(Output{ mOrigin, "DIGITS", 0, Lifetime::Timeframe }, digitAccum);
    // pc.outputs().snapshot(Output{ "FIT", "DIGITSMCTR", 0, Lifetime::Timeframe }, labelAccum);
    LOG(INFO) << "FIT: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{ mOrigin, "ROMode", 0, Lifetime::Timeframe }, mROMode);
    timer.Stop();
    LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  }

  // private:
 protected:
  Bool_t mContinuous = kFALSE;  ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns
  o2::detectors::DetID mID;
  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;
  Digitizer mT0Digitizer{ o2::t0::T0DigitizationParameters() }; ///< Digitizer
  //Digitizer mV0Digitizer; ///< Digitizer
  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode

  std::vector<TChain*> mSimChains;
};

//_______________________________________________
class FITT0DPLDigitizerTask : public FITDPLDigitizerTask
{
 public:
  // FIXME: origina should be extractable from the DetID, the problem is 3d party header dependencies
  static constexpr o2::detectors::DetID::ID DETID = o2::detectors::DetID::T0;
  static constexpr o2::header::DataOrigin DETOR = o2::header::gDataOriginT0;
  FITT0DPLDigitizerTask()
  {
    mID = DETID;
    mOrigin = DETOR;
  }
};

constexpr o2::detectors::DetID::ID FITT0DPLDigitizerTask::DETID;
constexpr o2::header::DataOrigin FITT0DPLDigitizerTask::DETOR;

o2::framework::DataProcessorSpec getFITT0DigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::string detStr = o2::detectors::DetID::getName(FITT0DPLDigitizerTask::DETID);
  auto detOrig = FITT0DPLDigitizerTask::DETOR;
  return DataProcessorSpec{ (detStr + "Digitizer").c_str(),
      Inputs{ InputSpec{ "collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe } },
      Outputs{ OutputSpec{ detOrig, "DIGITS", 0, Lifetime::Timeframe },
             /*OutputSpec{ detOrig "FIT", "DIGITSMCTR", 0, Lifetime::Timeframe }*/
             OutputSpec{  detOrig, "ROMode", 0, Lifetime::Timeframe } },
    AlgorithmSpec{ adaptFromTask<FITDPLDigitizerTask>() },
    Options{ { "simFile", VariantType::String, "o2sim.root", { "Sim (background) input filename" } },
             { "simFileS", VariantType::String, "", { "Sim (signal) input filename" } },
             { "pileup", VariantType::Int, 1, { "whether to run in continuous time mode" } } }
    // I can't use VariantType::Bool as it seems to have a problem
  };
}

} // end namespace fit
} // end namespace o2
