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
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/PedestalData.h"
#include "DataFormatsZDC/MCLabel.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCTruthContainer.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ZDCSimulation/Digitizer.h"
#include "ZDCSimulation/Detector.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "SimConfig/DigiParams.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace zdc
{

class ZDCDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
  using GRP = o2::parameters::GRPObject;

 public:
  ZDCDPLDigitizerTask() : o2::base::BaseDPLDigitizer(o2::base::InitServices::GEOM) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(INFO) << "Initializing ZDC digitization";

    auto& dopt = o2::conf::DigiParams::Instance();

    mDigitizer.setCCDBServer(dopt.ccdb);
    mDigitizer.init();
    mROMode = mDigitizer.isContinuous() ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::PRESENT;
  }

  void run(framework::ProcessingContext& pc)
  {
    LOG(INFO) << "Doing ZDC digitization";

    // TODO: this should eventually come from the framework and depend on the TF timestamp
    mDigitizer.refreshCCDB();

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(o2::detectors::DetID::ZDC, mSimChains);

    const auto& grp = context->getGRP();
    mDigitizer.setTimeStamp(grp.getTimeStart());
    if (mDigitizer.getNEmptyBunches() < 0) {
      mDigitizer.findEmptyBunches(context->getBunchFilling().getPattern());
    }
    auto& irecords = context->getEventRecords();
    auto& eventParts = context->getEventParts();
    if (irecords.empty()) {
      return;
    }
    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    std::vector<o2::zdc::Hit> hits;

    for (int collID = 0; collID < irecords.size(); ++collID) {

      const auto& irec = irecords[collID];
      mDigitizer.setInteractionRecord(irec);

      for (auto& part : eventParts[collID]) {

        context->retrieveHits(mSimChains, "ZDCHit", part.sourceID, part.entryID, &hits);
        LOG(INFO) << "For collision " << collID << " eventID " << part.entryID << " found ZDC " << hits.size() << " hits ";

        mDigitizer.setEventID(part.entryID);
        mDigitizer.setSrcID(part.sourceID);

        mDigitizer.process(hits, mDigitsBC, mDigitsCh, mLabels);
      }
    }

    o2::InteractionTimeRecord terminateIR;
    terminateIR.orbit = 0xffffffff; // supply IR in the infinite future to flush all cached BC
    mDigitizer.setInteractionRecord(terminateIR);
    mDigitizer.flush(mDigitsBC, mDigitsCh, mLabels);

    for (uint32_t ib = 0; ib < mDigitsBC.size(); ib++) {
      mDigitizer.assignTriggerBits(ib, mDigitsBC);
    }
    for (uint32_t ib = 0; ib < mDigitsBC.size(); ib++) {
      mDigitizer.maskTriggerBits(ib, mDigitsBC);
    }

    const auto &irFirst = irecords.front(), irLast = irecords.back();
    o2::InteractionRecord irPed(o2::constants::lhc::LHCMaxBunches - 1, irFirst.orbit);
    int norbits = irLast.orbit - irFirst.orbit + 1;
    mPedestalData.resize(norbits);
    for (int i = 0; i < norbits; i++) {
      mDigitizer.updatePedestalReference(mPedestalData[i]);
      mPedestalData[i].ir = irPed;
      irPed.orbit++;
    }

    // send out to next stage
    pc.outputs().snapshot(Output{"ZDC", "DIGITSBC", 0, Lifetime::Timeframe}, mDigitsBC);
    pc.outputs().snapshot(Output{"ZDC", "DIGITSCH", 0, Lifetime::Timeframe}, mDigitsCh);
    pc.outputs().snapshot(Output{"ZDC", "DIGITSPD", 0, Lifetime::Timeframe}, mPedestalData);
    if (pc.outputs().isAllowed({"ZDC", "DIGITSLBL", 0})) {
      pc.outputs().snapshot(Output{"ZDC", "DIGITSLBL", 0, Lifetime::Timeframe}, mLabels);
    }

    LOG(INFO) << "ZDC: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"ZDC", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

 private:
  Digitizer mDigitizer;
  std::vector<TChain*> mSimChains;
  std::vector<o2::zdc::ChannelData> mDigitsCh;
  std::vector<o2::zdc::BCData> mDigitsBC;
  std::vector<o2::zdc::PedestalData> mPedestalData;
  o2::dataformats::MCTruthContainer<o2::zdc::MCLabel> mLabels; // labels which get filled

  // RS: at the moment using hardcoded flag for continuous readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getZDCDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "DIGITSPD", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("ZDC", "DIGITSLBL", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("ZDC", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "ZDCDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},

    outputs,

    AlgorithmSpec{adaptFromTask<ZDCDPLDigitizerTask>()},
    Options{}};
}

} // end namespace zdc
} // end namespace o2
