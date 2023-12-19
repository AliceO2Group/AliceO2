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

#include "CTPDigitizerSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework//Task.h"
#include "DetectorsBase/BaseDPLDigitizer.h"
#include "DataFormatsCTP/Digits.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "DetectorsCommonDataFormats/DetID.h"
#include "CTPSimulation/Digitizer.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFV0/Digit.h"

#include <TStopwatch.h>
#include <gsl/span>

using namespace o2::framework;
namespace o2
{
namespace ctp
{
class CTPDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
  using GRP = o2::parameters::GRPObject;

 public:
  CTPDPLDigitizerTask(const std::vector<o2::detectors::DetID>& detList) : o2::base::BaseDPLDigitizer(), mDigitizer(), mDetList(detList) {}
  ~CTPDPLDigitizerTask() override = default;
  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDigitizer.init();
  }
  void run(framework::ProcessingContext& pc)
  {
    // read collision context from input
    // auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    // const bool withQED = context->isQEDProvided();
    // auto& timesview = context->getEventRecords(withQED);
    // read ctp inputs from input

    std::vector<o2::ctp::CTPInputDigit> finputs;
    TStopwatch timer;
    timer.Start();
    LOG(info) << "CALLING CTP DIGITIZATION";
    // Input order: T0, V0, ... but O need also position of inputs DETInputs
    // fv0
    if (std::find(mDetList.begin(), mDetList.end(), o2::detectors::DetID::FT0) != mDetList.end()) {
      auto ft0inputs = pc.inputs().get<gsl::span<o2::ft0::DetTrigInput>>("ft0");
      for (const auto& inp : ft0inputs) {
        finputs.emplace_back(CTPInputDigit{inp.mIntRecord, inp.mInputs, o2::detectors::DetID::FT0});
      }
    }
    // fv0
    if (std::find(mDetList.begin(), mDetList.end(), o2::detectors::DetID::FV0) != mDetList.end()) {
      auto fv0inputs = pc.inputs().get<gsl::span<o2::fv0::DetTrigInput>>("fv0");
      for (const auto& inp : fv0inputs) {
        finputs.emplace_back(CTPInputDigit{inp.mIntRecord, inp.mInputs, o2::detectors::DetID::FV0});
      }
    }
    // emc
    if (std::find(mDetList.begin(), mDetList.end(), o2::detectors::DetID::EMC) != mDetList.end()) {
      auto emcinputs = pc.inputs().get<gsl::span<o2::ctp::CTPInputDigit>>("emc");
      for (const auto& inp : emcinputs) {
        finputs.emplace_back(inp);
      }
    }
    gsl::span<CTPInputDigit> ginputs(finputs);
    auto digits = mDigitizer.process(ginputs);
    // send out to next stage
    LOG(info) << "CTP DIGITS being sent.";
    pc.outputs().snapshot(Output{"CTP", "DIGITS", 0, Lifetime::Timeframe}, digits);
    LOG(info) << "CTP PRESENT being sent.";
    pc.outputs().snapshot(Output{"CTP", "ROMode", 0, Lifetime::Timeframe}, mROMode);
    timer.Stop();
    LOG(info) << "CTP Digitization took " << timer.CpuTime() << "s";
  }

 protected:
  o2::parameters::GRPObject::ROMode mROMode = o2::parameters::GRPObject::PRESENT;
  o2::ctp::Digitizer mDigitizer; ///< Digitizer
  std::vector<o2::detectors::DetID> mDetList;
};
o2::framework::DataProcessorSpec getCTPDigitizerSpec(int channel, std::vector<o2::detectors::DetID>& detList, bool mctruth)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> output;

  if (std::find(detList.begin(), detList.end(), o2::detectors::DetID::FT0) != detList.end()) {
    inputs.emplace_back("ft0", "FT0", "TRIGGERINPUT", 0, Lifetime::Timeframe);
  }
  if (std::find(detList.begin(), detList.end(), o2::detectors::DetID::FV0) != detList.end()) {
    inputs.emplace_back("fv0", "FV0", "TRIGGERINPUT", 0, Lifetime::Timeframe);
  }
  if (std::find(detList.begin(), detList.end(), o2::detectors::DetID::EMC) != detList.end()) {
    inputs.emplace_back("emc", "EMC", "TRIGGERINPUT", 0, Lifetime::Timeframe);
  }
  output.emplace_back("CTP", "DIGITS", 0, Lifetime::Timeframe);
  output.emplace_back("CTP", "ROMode", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "CTPDigitizer",
    inputs,
    output,
    AlgorithmSpec{adaptFromTask<CTPDPLDigitizerTask>(detList)},
    Options{{"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}},
            {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}
} // namespace ctp
} // namespace o2
