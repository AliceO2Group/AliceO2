// V0.1
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   WriteRawFromDigitsSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to produce raw files from a Digits stream
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
#include "Framework/InputRecordWalker.h"

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
  LOG(INFO) << "[HMPID Write Raw File From Digits vector - init()]";
  mBaseFileName = ic.options().get<std::string>("out-file");
  mSkipEmpty = ic.options().get<bool>("skip-empty");
  mFixedPacketLenght = ic.options().get<bool>("fixed-lenght");
  mOrderTheEvents = ic.options().get<bool>("order-events");
  mDigitsReceived = 0;
  mFramesReceived = 0;

  mCod = new HmpidCoder(Geo::MAXEQUIPMENTS, mSkipEmpty, mFixedPacketLenght);
  mCod->reset();
  mCod->openOutputStream(mBaseFileName.c_str());

  mExTimer.start();
  return;
}

/*
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
*/

void WriteRawFromDigitsTask::run(framework::ProcessingContext& pc)
{
  for (auto const& ref : InputRecordWalker(pc.inputs())) {
    std::vector<o2::hmpid::Digit> digits = pc.inputs().get<std::vector<o2::hmpid::Digit>>(ref);
    if (mOrderTheEvents) {
      mDigits.insert(mDigits.end(), digits.begin(), digits.end());
    } else {
      mCod->addDigitsChunk(digits);
      mCod->codeDigitsChunk();
    }
    mDigitsReceived += digits.size();
    mFramesReceived++;
    LOG(DEBUG) << "run() Digits received =" << mDigitsReceived << " frames = " << mFramesReceived;
  }
  mExTimer.elapseMes("... Write raw file ... Digits received = " + std::to_string(mDigitsReceived) + " Frames received = " + std::to_string(mFramesReceived));
  return;
}

void WriteRawFromDigitsTask::endOfStream(framework::EndOfStreamContext& ec)
{
  mExTimer.logMes("Received an End Of Stream !");
  if (mOrderTheEvents) {
    //    sort(mDigits.begin(), mDigits.end(), eventEquipPadsComparision);
    sort(mDigits.begin(), mDigits.end(), o2::hmpid::Digit::eventEquipPadsComp);
    mExTimer.logMes("We sort " + std::to_string(mDigits.size()) + "  ! ");
    mCod->codeDigitsVector(mDigits);
  } else {
    mCod->codeDigitsChunk(true);
  }
  mCod->closeOutputStream();
  mCod->dumpResults();

  mExTimer.logMes("Raw File created ! Digits received = " + std::to_string(mDigitsReceived) + " Frame received =" + std::to_string(mFramesReceived));
  mExTimer.stop();
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getWriteRawFromDigitsSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("digits", o2::header::gDataOriginHMP, "DIGITS", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;

  return DataProcessorSpec{
    "HMP-WriteRawFromDigits",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<WriteRawFromDigitsTask>()},
    Options{{"out-file", VariantType::String, "hmpidRaw", {"name of the output file"}},
            {"order-events", VariantType::Bool, false, {"order the events time"}},
            {"skip-empty", VariantType::Bool, false, {"skip empty events"}},
            {"fixed-lenght", VariantType::Bool, false, {"fixed lenght packets = 8K bytes"}}}};
}

} // namespace hmpid
} // end namespace o2
