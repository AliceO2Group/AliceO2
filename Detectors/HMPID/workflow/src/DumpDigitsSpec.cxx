// draft
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DatDecoderSpec.cxx
/// \author  
///
/// \brief Implementation of a data processor to run the HMPID raw decoding
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "HMPIDBase/Digit.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDWorkflow/DumpDigitsSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void DumpDigitsTask::init(framework::InitContext& ic)
{

    return;
}

void DumpDigitsTask::run(framework::ProcessingContext& pc)
{

  LOG(INFO) << "[HMPID Dump Digits - run] Dumping ...";

//  auto digits = pc.inputs().get<OutputType>("digits");
//      mPDigits = &digits;

  for (auto&& input : pc.inputs()) {
 std::cout << input.spec->binding << std::endl;
    if (input.spec->binding == "digits") {
      const auto* header = o2::header::get<header::DataHeader*>(input.header);
      if (!header) {
        return;
      }
      std::vector<o2::hmpid::Digit>*theDigits = (std::vector<o2::hmpid::Digit>*)input.payload;
      std::cout << "--- HMP Digits : [Chamb,PhoCat,x,y]@(Orbit,BC)=Charge ---" << std::endl;
      for(o2::hmpid::Digit Dig : theDigits) {
        std::cout << Dig << std::endl;
      }
      std::cout << "---------------- HMP Dump Digits : EOF ------------------" << std::endl;
    }
  }





}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDumpDigitsSpec(std::string inputSpec)
//o2::framework::DataPrecessorSpec getDecodingSpec()
{
  
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("digits", o2::header::gDataOriginHMP, "DIGITS", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;

  
  return DataProcessorSpec{
    "HMP-DigitsDump",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DumpDigitsTask>()},
    Options{{"print", VariantType::Bool, false, {"print digits"}}} };
}

} // namespace hmpid
} // end namespace o2
