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
  LOG(INFO) << "[HMPID Dump Digits - run] Dumping ...";
  mPrintDigits = ic.options().get<bool>("print");

  mIsOutputOnFile = false;
  mOutputFileName = ic.options().get<std::string>("out-file");
  if(mOutputFileName != "") {
    mOsFile.open(mOutputFileName, std::ios::out);
    if (mOsFile.is_open()) {
      mIsOutputOnFile = true;
    }
  }
  return;
}

void DumpDigitsTask::run(framework::ProcessingContext& pc)
{
  //LOG(INFO) << "Enter Dump run...";

  for (auto&& input : pc.inputs()) {
    if (input.spec->binding == "digits") {
      auto digits = pc.inputs().get<std::vector<o2::hmpid::Digit>>("digits");
      LOG(INFO) << "The size of the vector =" << digits.size();
      if (mPrintDigits) {
         std::cout << "--- HMP Digits : [Chamb,PhoCat,x,y]@(Orbit,BC)=Charge ---" << std::endl;
         for(o2::hmpid::Digit Dig : digits) {
           std::cout << Dig << std::endl;
         }
         std::cout << "---------------- HMP Dump Digits : EOF ------------------" << std::endl;
      }
      if (mIsOutputOnFile) {
         mOsFile << "--- HMP Digits : [Chamb,PhoCat,x,y]@(Orbit,BC)=Charge ---" << std::endl;
         for(o2::hmpid::Digit Dig : digits) {
           mOsFile << Dig << std::endl;
         }
         mOsFile.close();
      }
    }
  }
  return;
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
    Options{{"out-file", VariantType::String, "", {"name of the output file"}},
            {"print", VariantType::Bool, false, {"print digits (default false )"}}} };
}

} // namespace hmpid
} // end namespace o2
