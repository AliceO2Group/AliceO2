// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MIDDigitizerSpec.h"
#include "TChain.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ChEffCounter.h"
#include "MIDSimulation/Digitizer.h"
#include "MIDSimulation/DigitsMerger.h"
#include "MIDSimulation/ChamberResponse.h"
#include "MIDSimulation/ChamberEfficiencyResponse.h"
#include "MIDSimulation/Geometry.h"
#include "DataFormatsMID/MCLabel.h"

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DPMAP = std::unordered_map<o2::dcs::DataPointIdentifier, std::vector<o2::dcs::DataPointValue>>;

namespace o2
{
namespace mid
{

class MIDDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
 public:
  MIDDPLDigitizerTask() : o2::base::BaseDPLDigitizer(o2::base::InitServices::GEOM) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    LOG(info) << "initializing MID digitization";

    mDigitizer = std::make_unique<Digitizer>(createDefaultChamberResponse(), createDefaultChamberEfficiencyResponse(), createTransformationFromManager(gGeoManager));
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
  {
    if (matcher == ConcreteDataMatcher(header::gDataOriginMID, "CHAMBER_EFF", 0)) {
      auto* chEffCounters = static_cast<std::vector<ChEffCounter>*>(obj);
      mDigitizer->setChamberEfficiency(*chEffCounters);
      return;
    }
    if (matcher == ConcreteDataMatcher(header::gDataOriginMID, "CHAMBER_HV", 0)) {
      auto* dpMap = static_cast<DPMAP*>(obj);
      mDigitizer->getChamberResponse().setHV(*dpMap);
      return;
    }
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }
    LOG(debug) << "Doing MID digitization";

    // read collision context from input
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    // Triggers reading from CCDB
    pc.inputs().get<std::vector<ChEffCounter>*>("mid_ch_eff");
    pc.inputs().get<DPMAP*>("mid_ch_hv");
    context->initSimChains(o2::detectors::DetID::MID, mSimChains);
    auto& irecords = context->getEventRecords();

    auto& eventParts = context->getEventParts();
    std::vector<o2::mid::ColumnData> digits, digitsAccum;
    std::vector<o2::mid::ROFRecord> rofRecords;
    o2::dataformats::MCTruthContainer<o2::mid::MCLabel> labels, labelsAccum;

    // loop over all composite collisions given from context
    // (aka loop over all the interaction records)
    for (int collID = 0; collID < irecords.size(); ++collID) {
      // for each collision, loop over the constituents event and source IDs
      // (background signal merging is basically taking place here)
      auto firstEntry = digitsAccum.size();
      for (auto& part : eventParts[collID]) {
        mDigitizer->setEventID(part.entryID);
        mDigitizer->setSrcID(part.sourceID);

        // get the hits for this event and this source
        std::vector<o2::mid::Hit> hits;
        context->retrieveHits(mSimChains, "MIDHit", part.sourceID, part.entryID, &hits);
        LOG(debug) << "For collision " << collID << " eventID " << part.entryID << " found MID " << hits.size() << " hits ";

        mDigitizer->process(hits, digits, labels);
        if (digits.empty()) {
          continue;
        }
        digitsAccum.insert(digitsAccum.end(), digits.begin(), digits.end());
        labelsAccum.mergeAtBack(labels);
      }
      auto nEntries = digitsAccum.size() - firstEntry;
      rofRecords.emplace_back(irecords[collID], EventType::Standard, firstEntry, nEntries);
    }

    mDigitsMerger.process(digitsAccum, labelsAccum, rofRecords);

    LOG(debug) << "MID: Sending " << digitsAccum.size() << " digits.";
    pc.outputs().snapshot(Output{"MID", "DIGITS", 0, Lifetime::Timeframe}, mDigitsMerger.getColumnData());
    pc.outputs().snapshot(Output{"MID", "DIGITSROF", 0, Lifetime::Timeframe}, mDigitsMerger.getROFRecords());
    if (pc.outputs().isAllowed({"MID", "DIGITLABELS", 0})) {
      pc.outputs().snapshot(Output{"MID", "DIGITLABELS", 0, Lifetime::Timeframe}, mDigitsMerger.getMCContainer());
    }
    LOG(debug) << "MID: Sending ROMode= " << mROMode << " to GRPUpdater";
    pc.outputs().snapshot(Output{"MID", "ROMode", 0, Lifetime::Timeframe}, mROMode);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    finished = true;
  }

 private:
  std::unique_ptr<Digitizer> mDigitizer;
  DigitsMerger mDigitsMerger;
  std::vector<TChain*> mSimChains;
  // RS: at the moment using hardcoded flag for continuos readout
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::CONTINUOUS; // readout mode
};

o2::framework::DataProcessorSpec getMIDDigitizerSpec(int channel, bool mctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)

  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back("collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe);
  inputSpecs.emplace_back("mid_ch_eff", header::gDataOriginMID, "CHAMBER_EFF", 0, Lifetime::Condition, ccdbParamSpec("MID/Calib/ChamberEfficiency"));
  inputSpecs.emplace_back("mid_ch_hv", header::gDataOriginMID, "CHAMBER_HV", 0, Lifetime::Condition, ccdbParamSpec("MID/Calib/HV"));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MID", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MID", "DIGITSROF", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("MID", "DIGITLABELS", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("MID", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "MIDDigitizer",
    inputSpecs,
    outputs,
    AlgorithmSpec{adaptFromTask<MIDDPLDigitizerTask>()},
    Options{}};
}

} // namespace mid
} // namespace o2
