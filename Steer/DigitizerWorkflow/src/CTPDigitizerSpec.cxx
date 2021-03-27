// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DataFormatsFV0/BCData.h"

#include <TStopwatch.h>

using namespace o2::framework;
namespace o2
{
namespace ctp
{
class CTPDPLDigitizerTask : public o2::base::BaseDPLDigitizer
{
  using GRP = o2::parameters::GRPObject;

 public:
  CTPDPLDigitizerTask() : o2::base::BaseDPLDigitizer(), mDigitizer() {}
  ~CTPDPLDigitizerTask() override = default;
  void initDigitizerTask(framework::InitContext& ic) override
  {
    mDigitizer.init();
  }
  void run(framework::ProcessingContext& pc)
  {
    if (mFinished) {
      return;
    }
    // read collision context from input
    //auto context = pc.inputs().get<o2::steer::DigitizationContext*>("collisioncontext");
    //const bool withQED = context->isQEDProvided();
    //auto& timesview = context->getEventRecords(withQED);
    // read ctp inputs from input
    auto ft0inputs = pc.inputs().get<std::vector<o2::ft0::DetTrigInput>>("ft0");
    auto fv0inputs = pc.inputs().get<std::vector<o2::fv0::DetTrigInput>>("fv0");

    // if there is nothing to do ... return
    if ((ft0inputs.size() == 0) && (fv0inputs.size() == 0)) {
      return;
    }
    std::map<o2::InteractionRecord, o2::ctp::CTPDigit> finputs;
    TStopwatch timer;
    timer.Start();
    LOG(INFO) << "CALLING CTP DIGITIZATION";
    //for (const auto& coll : timesview) {
    //  mDigitizer.setInteractionRecord(coll);
    //  mDigitizer.process(mDigits);
    //  mDigitizer.flush(mDigits);
    //}
    for (const auto& inp : ft0inputs) {
      CTPInputDigit finpdigit;
      finpdigit.mDetector = o2::detectors::DetID::FT0;
      finpdigit.mInputsMask = inp.mInputs;
      CTPDigit fctpdigit;
      fctpdigit.mIntRecord = inp.mIntRecord;
      fctpdigit.mInputs.push_back(finpdigit);
      finputs[inp.mIntRecord] = fctpdigit;
    }
    for (const auto& inp : fv0inputs) {
      CTPInputDigit finpdigit;
      finpdigit.mDetector = o2::detectors::DetID::FV0;
      finpdigit.mInputsMask = inp.mInputs;
      if (finputs.count(inp.mIntRecord) == 0) {
        CTPDigit fctpdigit;
        fctpdigit.mIntRecord = inp.mIntRecord;
        fctpdigit.mInputs.push_back(finpdigit);
        finputs[inp.mIntRecord] = fctpdigit;
      } else {
        finputs[inp.mIntRecord].mInputs.push_back(finpdigit);
      }
    }
    for (const auto& inps : finputs) {
      mDigitizer.setInteractionRecord(inps.first);
      mDigitizer.process(inps.second, mDigits);
      mDigitizer.flush(mDigits);
    }
    // send out to next stage
    pc.outputs().snapshot(Output{"CTP", "DIGITS", 0, Lifetime::Timeframe}, mDigits);
    timer.Stop();
    LOG(INFO) << "CTP Digitization took " << timer.CpuTime() << "s";
    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 protected:
  bool mFinished = false;
  std::vector<o2::ctp::CTPDigit> mDigits;

  Bool_t mContinuous = kFALSE;   ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1;  ///< Fair time unit in ns
  o2::ctp::Digitizer mDigitizer; ///< Digitizer
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
  output.emplace_back("CTP", "DIGITSBC", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "CTPDigitizer",
    inputs,
    output,
    AlgorithmSpec{adaptFromTask<CTPDPLDigitizerTask>()},
    Options{{"pileup", VariantType::Int, 1, {"whether to run in continuous time mode"}},
            {"disable-qed", o2::framework::VariantType::Bool, false, {"disable QED handling"}}}};
}
} // namespace ctp
} // namespace o2