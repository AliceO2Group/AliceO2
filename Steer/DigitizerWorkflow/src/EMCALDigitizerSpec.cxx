// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for RunContext
#include "TChain.h"

#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace emcal
{

void DigitizerSpec::init(framework::InitContext& ctx)
{
  // setup the input chain for the hits
  mSimChains.emplace_back(new TChain("o2sim"));

  // add the main (background) file
  mSimChains.back()->AddFile(ctx.options().get<std::string>("simFile").c_str());

  // maybe add a particular signal file
  auto signalfilename = ctx.options().get<std::string>("simFileS");
  if (signalfilename.size() > 0) {
    mSimChains.emplace_back(new TChain("o2sim"));
    mSimChains.back()->AddFile(signalfilename.c_str());
  }

  // make sure that the geometry is loaded (TODO will this be done centrally?)
  if (!gGeoManager) {
    o2::base::GeometryManager::loadGeometry();
  }
  // run 3 geometry == run 2 geometry for EMCAL
  // to be adapted with run numbers at a later stage
  auto geom = o2::emcal::Geometry::GetInstance("EMCAL_COMPLETE12SMV1_DCAL_8SM", "Geant4", "EMV-EMCAL");
  // init digitizer
  mDigitizer.setGeometry(geom);
  mDigitizer.init();

  mFinished = false;
}

void DigitizerSpec::run(framework::ProcessingContext& ctx)
{
  if (mFinished)
    return;

  // read collision context from input
  auto context = ctx.inputs().get<o2::steer::RunContext*>("collisioncontext");
  auto& timesview = context->getEventRecords();
  LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";

  // if there is nothing to do ... return
  if (timesview.size() == 0)
    return;

  TStopwatch timer;
  timer.Start();

  LOG(INFO) << " CALLING EMCAL DIGITIZATION ";
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> labelAccum;

  auto& eventParts = context->getEventParts();
  mAccumulatedDigits.clear();
  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < timesview.size(); ++collID) {
    mDigitizer.setEventTime(timesview[collID].timeNS);

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    for (auto& part : eventParts[collID]) {
      mDigitizer.setCurrEvID(part.entryID);
      mDigitizer.setCurrSrcID(part.sourceID);

      // get the hits for this event and this source
      mHits.clear();
      retrieveHits(mSimChains, "EMCHit", part.sourceID, part.entryID, &mHits);

      LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found " << mHits.size() << " hits ";

      // call actual digitization procedure
      mLabels.clear();
      mDigits.clear();
      mDigitizer.process(mHits, mDigits);
      // copy digits into accumulator
      std::copy(mDigits.begin(), mDigits.end(), std::back_inserter(mAccumulatedDigits));
      labelAccum.mergeAtBack(mLabels);
      LOG(INFO) << "Have " << mDigits.size() << " digits ";
    }
  }
  LOG(INFO) << "Have " << labelAccum.getNElements() << " EMCAL labels ";
  // here we have all digits and we can send them to consumer (aka snapshot it onto output)
  ctx.outputs().snapshot(Output{"EMC", "DIGITS", 0, Lifetime::Timeframe}, mAccumulatedDigits);
  ctx.outputs().snapshot(Output{"EMC", "DIGITSMCTR", 0, Lifetime::Timeframe}, labelAccum);
  // EMCAL is always a triggering detector
  const o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::TRIGGERING;
  LOG(INFO) << "EMCAL: Sending ROMode= " << roMode << " to GRPUpdater";
  ctx.outputs().snapshot(Output{"EMC", "ROMode", 0, Lifetime::Timeframe}, roMode);

  timer.Stop();
  LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";
  // we should be only called once; tell DPL that this process is ready to exit
  ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  mFinished = true;
}

void DigitizerSpec::retrieveHits(std::vector<TChain*> const& chains,
                                 const char* brname,
                                 int sourceID,
                                 int entryID,
                                 std::vector<Hit>* hits)
{
  auto br = chains[sourceID]->GetBranch(brname);
  if (!br) {
    LOG(ERROR) << "No branch found";
    return;
  }
  br->SetAddress(&hits);
  br->GetEntry(entryID);
}

DataProcessorSpec getEMCALDigitizerSpec(int channel)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  return DataProcessorSpec{
    "EMCALDigitizer", Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    Outputs{OutputSpec{"EMC", "DIGITS", 0, Lifetime::Timeframe},
            OutputSpec{"EMC", "DIGITSMCTR", 0, Lifetime::Timeframe},
            OutputSpec{"EMC", "ROMode", 0, Lifetime::Timeframe}},
    AlgorithmSpec{o2::framework::adaptFromTask<DigitizerSpec>()},
    Options{{"simFile", VariantType::String, "o2sim.root", {"Sim (background) input filename"}},
            {"simFileS", VariantType::String, "", {"Sim (signal) input filename"}},
            {"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}
    // I can't use VariantType::Bool as it seems to have a problem
  };
}
} // end namespace emcal
} // end namespace o2
