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
#include <algorithm>

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
#include "HMPIDSimulation/HmpidCoder.h"
#include "HMPIDWorkflow/WriteRawFromDigitsSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void WriteRawFromDigitsTask::init(framework::InitContext& ic)
{
  LOG(INFO) << "[HMPID Write Raw File From Digits vector - run] Dumping ...";
  mBaseFileName = ic.options().get<std::string>("out-file");


  return;
}

bool WriteRawFromDigitsTask::eventEquipPadsComparision(o2::hmpid::Digit d1, o2::hmpid::Digit d2)
{
  uint64_t t1,t2;
  t1 = d1.getTriggerID();
  t2 = d2.getTriggerID();

  if (t1 < t2) return true;
  if (t2 < t1) return false;

  if (d1.getPadID() < d2.getPadID()) return true;
  return false;
}


void WriteRawFromDigitsTask::run(framework::ProcessingContext& pc)
{
  for (auto&& input : pc.inputs()) {
    if (input.spec->binding == "digits") {
      auto digits = pc.inputs().get<std::vector<o2::hmpid::Digit>>("digits");

      LOG(INFO) << "The size of the digits vector " << digits.size();
      sort(digits.begin(), digits.end(), eventEquipPadsComparision);
      LOG(INFO) << "Digits sorted ! " ;

      HmpidCoder cod(Geo::MAXEQUIPMENTS);
      cod.openOutputStream(mBaseFileName.c_str());
      //cod.codeDigitsTest(2, 100);
      cod.codeDigitsVector(digits);
      cod.closeOutputStream();
      LOG(INFO) << "Raw File created ! " ;
    }
  }
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getWriteRawFromDigitsSpec(std::string inputSpec)
//o2::framework::DataPrecessorSpec getDecodingSpec()
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("digits", o2::header::gDataOriginHMP, "DIGITS", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  
  return DataProcessorSpec{
    "HMP-WriteRawFrtomDigits",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<WriteRawFromDigitsTask>()},
    Options{{"out-file", VariantType::String, "hmpidRaw", {"name of the output file"}}} };
}

} // namespace hmpid
} // end namespace o2
