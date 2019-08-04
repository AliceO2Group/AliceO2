// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Digitizer.h"
#include "MCHSimulation/Detector.h"
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
namespace mch
{

class MCHDPLDigitizerTask
{
 public:
  void init(framework::InitContext& ic)
  {
    LOG(DEBUG) << "initializing MCH digitization";
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

    if (!gGeoManager) {
      o2::base::GeometryManager::loadGeometry();
    }
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    LOG(DEBUG) << "Doing MCH digitization";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();

    for (auto& record : irecords) {
      LOG(DEBUG) << "MCH TIME RECEIVED " << record.timeNS;
    }

    auto& eventParts = context->getEventParts();
    std::vector<o2::mch::Digit> digitsAccum; // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum;

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < irecords.size(); ++collID) {
      mDigitizer.setEventTime(irecords[collID].timeNS);
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        mDigitizer.setEventID(part.entryID);
        mDigitizer.setSrcID(part.sourceID);

        // get the hits for this event and this source
        std::vector<o2::mch::Hit> hits;
        retrieveHits(mSimChains, "MCHHit", part.sourceID, part.entryID, &hits);
        LOG(DEBUG) << "For collision " << collID << " eventID " << part.entryID << " found MCH " << hits.size() << " hits ";

        std::vector<o2::mch::Digit> digits; // digits which get filled
        o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels;

        mDigitizer.process(hits, digits);
        mDigitizer.provideMC(labels);
        LOG(DEBUG) << "MCH obtained " << digits.size() << " digits ";
        for (auto& d : digits) {
          LOG(DEBUG) << "ADC " << d.getADC();
          LOG(DEBUG) << "PAD " << d.getPadID();
          LOG(DEBUG) << "TIME " << d.getTimeStamp();
          LOG(DEBUG) << "DetID " << d.getDetID();
        }
        std::copy(digits.begin(), digits.end(), std::back_inserter(digitsAccum));
        labelAccum.mergeAtBack(labels);
        LOG(DEBUG) << "labelAccum.getIndexedSize()  " << labelAccum.getIndexedSize();
        LOG(DEBUG) << "labelAccum.getNElements() " << labelAccum.getNElements();
        LOG(DEBUG) << "Have " << digits.size() << " digits ";
      }
    }
    mDigitizer.mergeDigits(digitsAccum, labelAccum);

    LOG(DEBUG) << "Have " << labelAccum.getNElements() << " MCH labels "; //does not work out!
    pc.outputs().snapshot(Output{"MCH", "DIGITS", 0, Lifetime::Timeframe}, digitsAccum);
    pc.outputs().snapshot(Output{"MCH", "DIGITSMCTR", 0, Lifetime::Timeframe}, labelAccum);
    LOG(DEBUG) << "MCH: Sending ROMode= " << mROMode << " to GRPUpdater";
    //ROMode: to be understood, check EMCal etc.
    pc.outputs().snapshot(Output{"MCH", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  }

 private:
  Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getMCHDigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "MCHDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    Outputs{OutputSpec{"MCH", "DIGITS", 0, Lifetime::Timeframe},
            OutputSpec{"MCH", "DIGITSMCTR", 0, Lifetime::Timeframe},
            OutputSpec{"MCH", "ROMode", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<MCHDPLDigitizerTask>()},
    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}}}};
}

} // end namespace mch
} // end namespace o2
