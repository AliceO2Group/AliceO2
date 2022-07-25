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

/// \file   DigitsToRootSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to Clusterize the Digits
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>

#include "HMPIDWorkflow/DigitsToClustersSpec.h"
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
#include "DataFormatsHMP/Cluster.h"
#include "HMPIDBase/Geo.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::header;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

// Splits a string in float array for string delimiter, TODO: Move this in a HMPID common library
void DigitsToClustersTask::strToFloatsSplit(std::string s, std::string delimiter, float* res, int maxElem)
{
  int index = 0;
  size_t pos_start = 0;
  size_t pos_end;
  size_t delim_len = delimiter.length();
  std::string token;
  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res[index++] = std::stof(token);
    if (index == maxElem) {
      return;
    }
  }
  res[index++] = (std::stof(s.substr(pos_start)));
}

//=======================
//
void DigitsToClustersTask::init(framework::InitContext& ic)
{
  LOG(info) << "[HMPID Clusterization - init() ] ";
  mSigmaCutPar = ic.options().get<std::string>("sigma-cut");
  if (mSigmaCutPar != "") {
    strToFloatsSplit(mSigmaCutPar, ",", mSigmaCut, 7);
  }

  mDigitsReceived = 0;
  mRec = new o2::hmpid::Clusterer();
  mExTimer.start();
}

void DigitsToClustersTask::run(framework::ProcessingContext& pc)
{
  LOG(debug) << "[HMPID DClusterization - run() ] Enter ...";
  auto triggers = pc.inputs().get<gsl::span<o2::hmpid::Trigger>>("intrecord");
  auto digits = pc.inputs().get<gsl::span<o2::hmpid::Digit>>("digits");
  std::vector<o2::hmpid::Cluster> clusters;
  std::vector<o2::hmpid::Trigger> clusterTriggers;

  for (const auto& trig : triggers) {
    if (trig.getNumberOfObjects()) {
      gsl::span<const o2::hmpid::Digit> trigDigits{digits.data() + trig.getFirstEntry(), size_t(trig.getNumberOfObjects())};
      size_t clStart = clusters.size();
      mRec->Dig2Clu(trigDigits, clusters, mSigmaCut, true);
      clusterTriggers.emplace_back(trig.getIr(), clStart, clusters.size() - clStart);
    }
  }
  LOGP(info, "Received {} triggers with {} digits -> {} triggers with {} clusters", triggers.size(), digits.size(), clusterTriggers.size(), clusters.size());
  mDigitsReceived += digits.size();

  pc.outputs().snapshot(o2::framework::Output{"HMP", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe}, clusters);
  pc.outputs().snapshot(o2::framework::Output{"HMP", "INTRECORDS1", 0, o2::framework::Lifetime::Timeframe}, clusterTriggers);

  mExTimer.elapseMes("Clusterization of Digits received = " + std::to_string(mDigitsReceived));
}

void DigitsToClustersTask::endOfStream(framework::EndOfStreamContext& ec)
{
  mExTimer.stop();
  mExTimer.logMes("End Clusterization !  digits = " + std::to_string(mDigitsReceived));
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDigitsToClustersSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("digits", o2::header::gDataOriginHMP, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("intrecord", o2::header::gDataOriginHMP, "INTRECORDS", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("HMP", "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("HMP", "INTRECORDS1", 0, o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "HMP-Clusterization",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DigitsToClustersTask>()},
    Options{{"sigma-cut", VariantType::String, "", {"sigmas as comma separated list"}}}};
}

} // namespace hmpid
} // end namespace o2
