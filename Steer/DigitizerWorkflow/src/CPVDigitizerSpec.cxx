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
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"

#include "CommonDataFormat/EvIndex.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "CPVSimulation/Digitizer.h"
#include "CPVBase/CPVSimParams.h"
#include "DataFormatsParameters/GRPObject.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "SimConfig/DigiParams.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace cpv
{

void DigitizerSpec::initDigitizerTask(framework::InitContext& ic)
{
  // init digitizer
  mDigitizer.init();

  if (mHits) {
    delete mHits;
  }
  mHits = new std::vector<Hit>();

  auto simulatePileup = ic.options().get<int>("pileup");
  if (simulatePileup) {                                              // set readout time and dead time parameters
    mReadoutTime = o2::cpv::CPVSimParams::Instance().mReadoutTimePU; //PHOS readout time in ns
    mDeadTime = o2::cpv::CPVSimParams::Instance().mDeadTimePU;       //PHOS dead time (should include readout => mReadoutTime< mDeadTime)
  } else {
    mReadoutTime = o2::cpv::CPVSimParams::Instance().mReadoutTime; //PHOS readout time in ns
    mDeadTime = o2::cpv::CPVSimParams::Instance().mDeadTime;       //PHOS dead time (should include readout => mReadoutTime< mDeadTime)
  }
}
// helper function which will be offered as a service
void DigitizerSpec::retrieveHits(const char* brname,
                                 int sourceID,
                                 int entryID)
{
  auto br = mSimChains[sourceID]->GetBranch(brname);
  if (!br) {
    LOG(ERROR) << "No branch found";
    return;
  }
  mHits->clear();
  br->SetAddress(&mHits);
  br->GetEntry(entryID);
}

void DigitizerSpec::run(framework::ProcessingContext& pc)
{

  // read collision context from input
  auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
  context->initSimChains(o2::detectors::DetID::CPV, mSimChains);
  auto& timesview = context->getEventRecords();
  LOG(DEBUG) << "GOT " << timesview.size() << " COLLISSION TIMES";

  // if there is nothing to do ... return
  int n = timesview.size();
  if (n == 0) {
    return;
  }

  TStopwatch timer;
  timer.Start();

  LOG(INFO) << " CALLING CPV DIGITIZATION ";
  std::vector<TriggerRecord> triggers;

  int indexStart = mDigitsOut.size();
  auto& eventParts = context->getEventParts();
  //if this is last stream of hits and we can write directly to final vector of digits? Otherwize use temporary vectors
  bool isLastStream = true;
  double eventTime = timesview[0].getTimeNS() - o2::cpv::CPVSimParams::Instance().mDeadTime; //checked above that list not empty
  int eventId;
  // loop over all composite collisions given from context
  // (aka loop over all the interaction records)
  for (int collID = 0; collID < n; ++collID) {

    double dt = timesview[collID].getTimeNS() - eventTime; //start new PHOS readout, continue current or dead time?
    if (dt > mReadoutTime && dt < mDeadTime) {             //dead time, skip event
      continue;
    }

    if (dt >= o2::cpv::CPVSimParams::Instance().mDeadTime) { // start new event
      //new event
      eventTime = timesview[collID].getTimeNS();
      dt = 0.;
      eventId = collID;
    }

    //Check if next event has to be added to this read-out
    if (collID < n - 1) {
      isLastStream = (timesview[collID + 1].getTimeNS() - eventTime > mReadoutTime);
    } else {
      isLastStream = true;
    }

    // for each collision, loop over the constituents event and source IDs
    // (background signal merging is basically taking place here)
    // merge new hist to current digit list
    auto part = eventParts[collID].begin();
    while (part != eventParts[collID].end()) {
      // get the hits for this event and this source
      int source = part->sourceID;
      int entry = part->entryID;
      retrieveHits("CPVHit", source, entry);
      part++;
      if (part == eventParts[collID].end() && isLastStream) { //last stream, copy digits directly to output vector
        mDigitizer.processHits(mHits, mDigitsFinal, mDigitsOut, mLabels, collID, source, dt);
        mDigitsFinal.clear();
        //finalyze previous event and clean
        // Add trigger record
        triggers.emplace_back(timesview[eventId], indexStart, mDigitsOut.size() - indexStart);
        indexStart = mDigitsOut.size();
        mDigitsFinal.clear();
      } else { //Fill intermediate digitvector
        mDigitsTmp.swap(mDigitsFinal);
        mDigitizer.processHits(mHits, mDigitsTmp, mDigitsFinal, mLabels, collID, source, dt);
        mDigitsTmp.clear();
      }
    }
  }
  LOG(DEBUG) << "Have " << mLabels.getNElements() << " CPV labels ";
  // here we have all digits and we can send them to consumer (aka snapshot it onto output)
  pc.outputs().snapshot(Output{"CPV", "DIGITS", 0, Lifetime::Timeframe}, mDigitsOut);
  pc.outputs().snapshot(Output{"CPV", "DIGITTRIGREC", 0, Lifetime::Timeframe}, triggers);
  if (pc.outputs().isAllowed({"CPV", "DIGITSMCTR", 0})) {
    pc.outputs().snapshot(Output{"CPV", "DIGITSMCTR", 0, Lifetime::Timeframe}, mLabels);
  }
  // CPV is always a triggered detector
  const o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::TRIGGERING;
  LOG(DEBUG) << "CPV: Sending ROMode= " << roMode << " to GRPUpdater";
  pc.outputs().snapshot(Output{"CPV", "ROMode", 0, Lifetime::Timeframe}, roMode);

  timer.Stop();
  LOG(INFO) << "Digitization took " << timer.CpuTime() << "s";

  //  pc.services().get<o2::framework::ControlService>().endOfStream();
  // we should be only called once; tell DPL that this process is ready to exit
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getCPVDigitizerSpec(int channel, bool mctruth)
{

  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("CPV", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("CPV", "DIGITTRIGREC", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("CPV", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("CPV", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "CPVDigitizer", Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{o2::framework::adaptFromTask<DigitizerSpec>()},
    Options{{"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}}}};
}
} // namespace cpv
} // namespace o2
