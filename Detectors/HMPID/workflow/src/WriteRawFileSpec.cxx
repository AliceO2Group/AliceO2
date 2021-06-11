// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   WriteRawFileSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to produce raw files from a Digits stream
///

#include "HMPIDWorkflow/WriteRawFileSpec.h"

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
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "CommonDataFormat/InteractionRecord.h"

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDSimulation/HmpidCoder2.h"

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
void WriteRawFileTask::init(framework::InitContext& ic)
{
  LOG(INFO) << "[HMPID Write Raw File From Digits vector - init()]";
  mBaseFileName = ic.options().get<std::string>("out-file");
  mSkipEmpty = ic.options().get<bool>("skip-empty");
  mFixedPacketLenght = ic.options().get<bool>("fixed-lenght");
  mOrderTheEvents = ic.options().get<bool>("order-events");
  mDigitsReceived = 0;
  mFramesReceived = 0;

  mCod = new HmpidCoder2(Geo::MAXEQUIPMENTS);
  mCod->setSkipEmptyEvents(mSkipEmpty);
  mCod->openOutputStream(mBaseFileName.c_str(), "all");

  mExTimer.start();
  return;
}

void WriteRawFileTask::run(framework::ProcessingContext& pc)
{
  std::vector<o2::hmpid::Trigger> triggers;
  std::vector<o2::hmpid::Digit> digits;

  for (auto const& ref : InputRecordWalker(pc.inputs())) {
    if (DataRefUtils::match(ref, {"check", ConcreteDataTypeMatcher{gDataOriginHMP, "INTRECORDS"}})) {
      triggers = pc.inputs().get<std::vector<o2::hmpid::Trigger>>(ref);
    }
    if (DataRefUtils::match(ref, {"check", ConcreteDataTypeMatcher{gDataOriginHMP, "DIGITS"}})) {
      digits = pc.inputs().get<std::vector<o2::hmpid::Digit>>(ref);
      LOG(DEBUG) << "The size of the vector =" << digits.size();
    }
  }

  for (int i = 0; i < triggers.size(); i++) {
    if (mOrderTheEvents) {
      int first = mDigits.size();
      mDigits.insert(mDigits.end(), digits.begin() + triggers[i].getFirstEntry(), digits.begin() + triggers[i].getLastEntry());
      mEvents.push_back({triggers[i].getIr(), first, int(mDigits.size() - first)});
    } else {
      std::vector<o2::hmpid::Digit> dig = {digits.begin() + triggers[i].getFirstEntry(), digits.begin() + triggers[i].getLastEntry()};
      mCod->codeEventChunkDigits(dig, triggers[i].getIr());
    }
  }
  mDigitsReceived += digits.size();
  mFramesReceived++;
  LOG(DEBUG) << "run() Digits received =" << mDigitsReceived << " frames = " << mFramesReceived;

  mExTimer.elapseMes("Write raw file ... Digits received = " + std::to_string(mDigitsReceived) + " Frames received = " + std::to_string(mFramesReceived));
  return;
}

void WriteRawFileTask::endOfStream(framework::EndOfStreamContext& ec)
{
  std::vector<o2::hmpid::Digit> dig;

  mExTimer.logMes("Received an End Of Stream !");
  if (mOrderTheEvents && mEvents.size() > 0) {
    sort(mEvents.begin(), mEvents.end());
    uint32_t orbit = mEvents[0].getOrbit();
    uint16_t bc = mEvents[0].getBc();
    for (int idx = 0; idx < mEvents.size(); idx++) {
      if (mSkipEmpty && (mEvents[idx].getNumberOfObjects() == 0 || mEvents[idx].getOrbit() == 0)) {
        continue;
      }
      if (mEvents[idx].getOrbit() != orbit || mEvents[idx].getBc() != bc) {
        mCod->codeEventChunkDigits(dig, o2::InteractionRecord(bc, orbit));
        LOG(INFO) << " Event :" << idx << " orbit=" << orbit << " bc=" << bc << " Digits:" << dig.size();
        dig.clear();
        orbit = mEvents[idx].getOrbit();
        bc = mEvents[idx].getBc();
      }
      for (int i = mEvents[idx].getFirstEntry(); i <= mEvents[idx].getLastEntry(); i++) {
        dig.push_back(mDigits[i]);
      }
    }
    mCod->codeEventChunkDigits(dig, o2::InteractionRecord(bc, orbit));
    LOG(INFO) << " Event :" << mEvents.size() - 1 << " orbit=" << orbit << " bc=" << bc << " Digits:" << dig.size();
  }
  mCod->closeOutputStream();
  mCod->dumpResults("all");

  mExTimer.logMes("Raw File created ! Digits received = " + std::to_string(mDigitsReceived) + " Frame received =" + std::to_string(mFramesReceived));
  mExTimer.stop();
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getWriteRawFileSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("digits", o2::header::gDataOriginHMP, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("intrecord", o2::header::gDataOriginHMP, "INTRECORDS", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;

  return DataProcessorSpec{
    "HMP-WriteRawFile",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<WriteRawFileTask>()},
    Options{{"out-file", VariantType::String, "hmpidRaw", {"name of the output file"}},
            {"order-events", VariantType::Bool, false, {"order the events time"}},
            {"skip-empty", VariantType::Bool, false, {"skip empty events"}},
            {"fixed-lenght", VariantType::Bool, false, {"fixed lenght packets = 8K bytes"}}}};
}

} // namespace hmpid
} // end namespace o2
