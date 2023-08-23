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

#include "MCHDigitizerSpec.h"

#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "MCHDigitFiltering/DigitFilterParam.h"
#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Digitizer.h"
#include "MCHSimulation/DigitizerParam.h"
#include "MCHSimulation/Hit.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include <TGeoManager.h>
#include <algorithm>
#include <map>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

namespace o2
{
namespace mch
{

class MCHDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
 public:
  MCHDPLDigitizerTask() : o2::base::BaseDPLDigitizer(o2::base::InitServices::GEOM) {}

  void initDigitizerTask(framework::InitContext& ic) override
  {
    auto transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
    mDigitizer = std::make_unique<Digitizer>(transformation);
  }

  void logStatus(gsl::span<Digit> digits, gsl::span<ROFRecord> rofs, o2::dataformats::MCLabelContainer& labels,
                 size_t nPileup, std::chrono::high_resolution_clock::time_point start)
  {
    LOGP(info, "Number of digits : {}", digits.size());
    LOGP(info, "Number of rofs : {}", rofs.size());
    LOGP(info, "Number of labels : {} (indexed {})", labels.getNElements(), labels.getIndexedSize());
    if (labels.getIndexedSize() != digits.size()) {
      LOGP(error, "Number of labels != number of digits");
    }
    LOGP(info, "Number of signal pileup : {} ({} %)", nPileup, 100. * nPileup / digits.size());
    auto tEnd = std::chrono::high_resolution_clock::now();
    auto duration = tEnd - start;
    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    LOGP(info, "Digitizer time {} ms", d);
  }

  void run(framework::ProcessingContext& pc)
  {
    static bool finished = false;
    if (finished) {
      return;
    }

    auto tStart = std::chrono::high_resolution_clock::now();
    auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    context->initSimChains(o2::detectors::DetID::MCH, mSimChains);
    const auto& eventRecords = context->getEventRecords();
    auto timeOffset = DigitFilterParam::Instance().timeOffset;

    // generate signals produced by every hits
    if (!DigitizerParam::Instance().onlyNoise) {
      const auto& eventParts = context->getEventParts();
      for (auto i = 0; i < eventRecords.size(); i++) {
        auto ir = eventRecords[i];
        // apply time offset, discarding events going to negative IR
        if (ir.toLong() < timeOffset) {
          continue;
        }
        ir -= timeOffset;
        for (const auto& part : eventParts[i]) {
          std::vector<Hit> hits{};
          context->retrieveHits(mSimChains, "MCHHit", part.sourceID, part.entryID, &hits);
          mDigitizer->processHits(hits, ir, part.entryID, part.sourceID);
        }
      }
    }

    // generate noise-only signals between first and last collisions ± 100 BC (= 25 ADC samples)
    auto firstIR = InteractionRecord::long2IR(std::max(int64_t(0), eventRecords.front().toLong() - timeOffset - 100));
    auto lastIR = InteractionRecord::long2IR(std::max(int64_t(0), eventRecords.back().toLong() - timeOffset + 100));
    mDigitizer->addNoise(firstIR, lastIR);

    // digitize
    std::vector<Digit> digits{};
    std::vector<ROFRecord> rofs{};
    dataformats::MCLabelContainer labels{};
    auto nPileup = mDigitizer->digitize(rofs, digits, labels);

    pc.outputs().snapshot(Output{"MCH", "DIGITS", 0, Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(Output{"MCH", "DIGITROFS", 0, Lifetime::Timeframe}, rofs);
    if (pc.outputs().isAllowed({"MCH", "DIGITSLABELS", 0})) {
      pc.outputs().snapshot(Output{"MCH", "DIGITSLABELS", 0, Lifetime::Timeframe}, labels);
    }
    pc.outputs().snapshot(Output{"MCH", "ROMode", 0, Lifetime::Timeframe},
                          DigitizerParam::Instance().continuous ? o2::parameters::GRPObject::CONTINUOUS : o2::parameters::GRPObject::TRIGGERING);

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    finished = true;

    logStatus(digits, rofs, labels, nPileup, tStart);
  }

 private:
  std::unique_ptr<Digitizer> mDigitizer{};
  std::vector<TChain*> mSimChains{};
};

o2::framework::DataProcessorSpec getMCHDigitizerSpec(int channel, bool mctruth)
{
  std::vector<OutputSpec> outputs{};
  outputs.emplace_back("MCH", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MCH", "DIGITROFS", 0, Lifetime::Timeframe);
  if (mctruth) {
    outputs.emplace_back("MCH", "DIGITSLABELS", 0, Lifetime::Timeframe);
  }
  outputs.emplace_back("MCH", "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "MCHDigitizer",
    Inputs{InputSpec{"collisioncontext", "SIM", "COLLISIONCONTEXT", static_cast<SubSpecificationType>(channel), Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<MCHDPLDigitizerTask>()},
    Options{}};
}

} // end namespace mch
} // end namespace o2
