// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   DumpDigitsSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to Dump o record a stream of Digits
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
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDWorkflow/DumpDigitsSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::header;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void DumpDigitsTask::init(framework::InitContext& ic)
{
  LOG(INFO) << "[HMPID Dump Digits - init() ] ";
  mPrintDigits = ic.options().get<bool>("print");

  mIsOutputOnFile = false;
  mOutputFileName = ic.options().get<std::string>("out-file");
  if (mOutputFileName != "") {
    mOsFile.open(mOutputFileName, std::ios::out);
    if (mOsFile.is_open()) {
      mIsOutputOnFile = true;
    }
  }
  if (mPrintDigits) {
    std::cout << "--- HMP Digits : [Chamb,PhoCat,x,y]=Charge ---" << std::endl;
  }
  if (mIsOutputOnFile) {
    mOsFile << "--- HMP Digits : [Chamb,PhoCat,x,y]=Charge ---" << std::endl;
  }
  mOrbit = -1;
  mBc = -1;
  mDigitsReceived = 0;

  mExTimer.start();
  return;
}

void DumpDigitsTask::run(framework::ProcessingContext& pc)
{
  LOG(INFO) << "[HMPID Dump Digits - run() ] Enter Dump ...";
  std::vector<o2::hmpid::Trigger> triggers;
  std::vector<o2::hmpid::Digit> digits;

  for (auto const& ref : InputRecordWalker(pc.inputs())) {
    if (DataRefUtils::match(ref, {"check", ConcreteDataTypeMatcher{gDataOriginHMP, "INTRECORDS"}})) {
      triggers = pc.inputs().get<std::vector<o2::hmpid::Trigger>>(ref);
      LOG(INFO) << "We receive triggers =" << triggers.size();
    }
    if (DataRefUtils::match(ref, {"check", ConcreteDataTypeMatcher{gDataOriginHMP, "DIGITS"}})) {
      digits = pc.inputs().get<std::vector<o2::hmpid::Digit>>(ref);
      LOG(INFO) << "The size of the vector =" << digits.size();
      mDigitsReceived += digits.size();
    }
    for (int i = 0; i < triggers.size(); i++) {
      if (mPrintDigits) {
        std::cout << "Trigger Event     Orbit = " << triggers[i].getOrbit() << "  BC = " << triggers[i].getBc() << std::endl;
        for (int j = triggers[i].getFirstEntry(); j <= triggers[i].getLastEntry(); j++) {
          std::cout << digits[j] << std::endl;
        }
      }
      if (mIsOutputOnFile) {
        mOsFile << "Trigger Event     Orbit = " << triggers[i].getOrbit() << "  BC = " << triggers[i].getBc() << std::endl;
        for (int j = triggers[i].getFirstEntry(); j <= triggers[i].getLastEntry(); j++) {
          mOsFile << digits[j] << std::endl;
        }
      }
    }
  }
  std::cout << ">>----->>>" << triggers.size() << "  " << digits.size() << std::endl;
  mExTimer.elapseMes("Dumping Digits received = " + std::to_string(mDigitsReceived));
  return;
}

void DumpDigitsTask::endOfStream(framework::EndOfStreamContext& ec)
{
  mOsFile.close();

  mExTimer.stop();
  mExTimer.logMes("End Digits Dump ! Dumped digits = " + std::to_string(mDigitsReceived));
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDumpDigitsSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("digits", o2::header::gDataOriginHMP, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("intrecord", o2::header::gDataOriginHMP, "INTRECORDS", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;

  return DataProcessorSpec{
    "HMP-DigitsDump",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DumpDigitsTask>()},
    Options{{"out-file", VariantType::String, "", {"name of the output file"}},
            {"print", VariantType::Bool, false, {"print digits (default false )"}}}};
}

} // namespace hmpid
} // end namespace o2
