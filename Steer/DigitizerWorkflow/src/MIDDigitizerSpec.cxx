// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MIDDigitizerSpec.h"
#include "TChain.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "DetectorsBase/GeometryManager.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/Digitizer.h"
#include "MIDSimulation/ChamberResponse.h"
#include "MIDSimulation/ChamberEfficiencyResponse.h"
#include "MIDSimulation/Geometry.h"
#include "MIDSimulation/MCLabel.h"

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
namespace mid
{

class MIDDPLDigitizerTask
{
 public:
  // MIDDPLDigitizerTask(Digitizer digitizer) : mDigitizer(nullptr)
  // {
  //   /// Ctor
  // }

  void init(framework::InitContext& ic)
  {
    LOG(INFO) << "initializing MID digitization";
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

    mDigitizer = std::make_unique<Digitizer>(createDefaultChamberResponse(), createDefaultChamberEfficiencyResponse(), createTransformationFromManager(gGeoManager));
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    LOG(INFO) << "Doing MID digitization";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
    auto& irecords = context->getEventRecords();

    for (auto& record : irecords) {
      LOG(INFO) << "MID TIME RECEIVED " << record.timeNS;
    }

    auto& eventParts = context->getEventParts();
    std::vector<o2::mid::ColumnDataMC> digitsAccum; // accumulator for digits
    o2::dataformats::MCTruthContainer<o2::mid::MCLabel> labelsAccum;

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < irecords.size(); ++collID) {
      mDigitizer->setEventTime(irecords[collID].timeNS);

      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      for (auto& part : eventParts[collID]) {
        mDigitizer->setEventID(part.entryID);
        mDigitizer->setSrcID(part.sourceID);

        // get the hits for this event and this source
        std::vector<o2::mid::Hit> hits;
        retrieveHits(mSimChains, "MIDHit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found MID " << hits.size() << " hits ";

        std::vector<o2::mid::ColumnDataMC> digits; // digits which get filled
        o2::dataformats::MCTruthContainer<o2::mid::MCLabel> labels;
        mDigitizer->process(hits, digits, labels);
        std::copy(digits.begin(), digits.end(), std::back_inserter(digitsAccum));
        labelsAccum.mergeAtBack(labels);
      }
    }

    LOG(INFO) << "MID: Sending " << digitsAccum.size() << " digits";
    pc.outputs().snapshot(Output{"MID", "DIGITS", 0, Lifetime::Timeframe}, digitsAccum);
    pc.outputs().snapshot(Output{"MID", "DIGITSMC", 0, Lifetime::Timeframe}, labelsAccum);

    LOG(INFO) << "MID: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"MID", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(false);
    finished = true;
  }

 private:
  std::unique_ptr<Digitizer> mDigitizer;
  std::vector<TChain*> mSimChains;
  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getMIDDigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "MIDDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    Outputs{OutputSpec{"MID", "DIGITS", 0, Lifetime::Timeframe},
            OutputSpec{"MID", "DIGITSMC", 0, Lifetime::Timeframe},
            OutputSpec{"MID", "ROMode", 0, Lifetime::Timeframe}},

    AlgorithmSpec{adaptFromTask<MIDDPLDigitizerTask>()},

    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}}}};
}

} // namespace mid
} // namespace o2
