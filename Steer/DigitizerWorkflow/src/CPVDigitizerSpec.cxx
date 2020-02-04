// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVDigitizerSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"

#include "CommonDataFormat/EvIndex.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "CPVSimulation/Digitizer.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsBase/GeometryManager.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace cpv
{

void DigitizerSpec::init(framework::InitContext& ic)
{

  // setup the input chain for the hits
  if (mSimChain) {
    delete mSimChain;
  }
  mSimChain = new TChain("o2sim");

  // add the main (background) file
  mSimChain->AddFile(ic.options().get<std::string>("simFile").c_str());

  // maybe add a particular signal file
  auto signalfilename = ic.options().get<std::string>("simFileS");
  if (signalfilename.size() > 0) {
    mSimChainS = new TChain("o2sim");
    mSimChainS->AddFile(signalfilename.c_str());
  }

  // make sure that the geometry is loaded (TODO will this be done centrally?)
  if (!gGeoManager) {
    o2::base::GeometryManager::loadGeometry();
  }
  // init digitizer
  mDigitizer.init();

  if (mHitsS) {
    delete mHitsS;
  }
  mHitsS = new std::vector<Hit>();
  if (mHitsBg) {
    delete mHitsBg;
  }
  mHitsBg = new std::vector<Hit>();

  mFinished = false;
}
// helper function which will be offered as a service
void DigitizerSpec::retrieveHits(const char* brname,
                                 int sourceID,
                                 int entryID)
{

  if (sourceID == 0) { //Bg
    mHitsBg->clear();
    auto br = mSimChain->GetBranch(brname);
    if (!br) {
      LOG(ERROR) << "No branch found";
      return;
    }
    br->SetAddress(&mHitsBg);
    br->GetEntry(entryID);
  } else { //Bg
    mHitsS->clear();
    auto br = mSimChainS->GetBranch(brname);
    if (!br) {
      LOG(ERROR) << "No branch found";
      return;
    }
    br->SetAddress(&mHitsS);
    br->GetEntry(entryID);
  }
}

void DigitizerSpec::run(framework::ProcessingContext& pc)
{
  if (mFinished)
    return;

  // read collision context from input
  auto context = pc.inputs().get<o2::steer::RunContext*>("collisioncontext");
  auto& timesview = context->getEventRecords();
  LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";

  // if there is nothing to do ... return
  if (timesview.size() == 0) {
    return;
  }

  TStopwatch timer;
  timer.Start();

  LOG(INFO) << " CALLING CPV DIGITIZATION ";
  std::vector<TriggerRecord> triggers;
  static std::vector<o2::cpv::Hit> hits;

  mLabels.clear();
  mDigits.clear();
  int indexStart = mDigits.size();
  auto& eventParts = context->getEventParts();
  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < timesview.size(); ++collID) {
    mDigitizer.setEventTime(timesview[collID].timeNS);

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {

      // get the hits for this event and this source
      retrieveHits("CPVHit", part.sourceID, part.entryID);
      mDigitizer.setCurrEvID(part.entryID);
    }

    LOG(DEBUG) << "Found " << mHitsBg->size() << " BG hits and " << mHitsS->size() << "signal hits";

    // call actual digitization procedure
    mDigitizer.process(mHitsBg, mHitsS, mDigits, mLabels);

    // Add trigger record
    triggers.emplace_back(timesview[collID], indexStart, mDigits.size() - indexStart);
    indexStart = mDigits.size();

    LOG(DEBUG) << "Have " << mDigits.size() << " digits ";
  }

  LOG(DEBUG) << "Have " << mLabels.getNElements() << " CPV labels ";
  // here we have all digits and we can send them to consumer (aka snapshot it onto output)
  pc.outputs().snapshot(Output{"CPV", "DIGITS", 0, Lifetime::Timeframe}, mDigits);
  pc.outputs().snapshot(Output{"CPV", "DIGITTRIGREC", 0, Lifetime::Timeframe}, triggers);
  pc.outputs().snapshot(Output{"CPV", "DIGITSMCTR", 0, Lifetime::Timeframe}, mLabels);
  // CPV is always a triggering detector
  const o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::TRIGGERING;
  LOG(DEBUG) << "CPV: Sending ROMode= " << roMode << " to GRPUpdater";
  pc.outputs().snapshot(Output{"CPV", "ROMode", 0, Lifetime::Timeframe}, roMode);

  timer.Stop();
  LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

  pc.services().get<o2::framework::ControlService>().endOfStream();
  // we should be only called once; tell DPL that this process is ready to exit
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  mFinished = true;
}

DataProcessorSpec getCPVDigitizerSpec(int channel)
{

  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "CPVDigitizer", Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    Outputs{OutputSpec{"CPV", "DIGITS", 0, Lifetime::Timeframe},
            OutputSpec{"CPV", "DIGITTRIGREC", 0, Lifetime::Timeframe},
            OutputSpec{"CPV", "DIGITSMCTR", 0, Lifetime::Timeframe},
            OutputSpec{"CPV", "ROMode", 0, Lifetime::Timeframe}},
    AlgorithmSpec{o2::framework::adaptFromTask<DigitizerSpec>()},
    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
            {"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}};
}
} // namespace cpv
} // namespace o2
