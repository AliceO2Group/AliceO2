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

#include <string>
#include <fairlogger/Logger.h>
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsRaw/RDHUtils.h"
#include "CTPWorkflow/RawDecoderSpec.h"
#include "CommonUtils/VerbosityConfig.h"

using namespace o2::ctp::reco_workflow;

void RawDecoderSpec::init(framework::InitContext& ctx)
{
  mNTFToIntegrate = ctx.options().get<int>("ntf-to-average");
  mVerbose = ctx.options().get<bool>("use-verbose-mode");
  mDecoder.setVerbose(mVerbose);
  mDecoder.setDoLumi(mDoLumi);
  mDecoder.setDoDigits(mDoDigits);
  LOG(info) << "CTP reco init done";
}
void RawDecoderSpec::endOfStream(framework::EndOfStreamContext& ec)
{
  auto& TFOrbits = mDecoder.getTFOrbits();
  std::sort(TFOrbits.begin(), TFOrbits.end());
  size_t l = TFOrbits.size();
  uint32_t o0 = 0;
  if (l) {
    o0 = TFOrbits[0];
  }
  int nmiss = 0;
  int nprt = 0;
  std::cout << "Missing orbits:";
  for (int i = 1; i < l; i++) {
    if ((TFOrbits[i] - o0) > 0x20) {
      if (nprt < 20) {
        std::cout << " " << o0 << "-" << TFOrbits[i];
      }
      nmiss += (TFOrbits[i] - o0) / 0x20;
      nprt++;
    }
    o0 = TFOrbits[i];
  }
  std::cout << std::endl;
  std::cout << "Number of missing TF:" << nmiss << std::endl;
}
void RawDecoderSpec::run(framework::ProcessingContext& ctx)
{
  mOutputDigits.clear();
  std::map<o2::InteractionRecord, CTPDigit> digits;
  using InputSpec = o2::framework::InputSpec;
  using ConcreteDataTypeMatcher = o2::framework::ConcreteDataTypeMatcher;
  using Lifetime = o2::framework::Lifetime;
  // setUpDummyLink
  auto& inputs = ctx.inputs();
  auto dummyOutput = [&ctx, this]() {
    if (this->mDoDigits) {
      ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, this->mOutputDigits);
    }
    if (this->mDoLumi) {
      ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}, this->mOutputLumiInfo);
    }
  };
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", o2::framework::ConcreteDataMatcher{"CTP", "RAWDATA", 0xDEADBEEF}}};
    for (const auto& ref : o2::framework::InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        dummyOutput();
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }
  //
  std::vector<LumiInfo> lumiPointsHBF1;
  int ret = mDecoder.decodeRaw(inputs, digits, lumiPointsHBF1);
  if (ret == 1) {
    dummyOutput();
    return;
  }
  if (mDoDigits) {
    for (auto const digmap : digits) {
      mOutputDigits.push_back(digmap.second);
    }
    LOG(info) << "[CTPRawToDigitConverter - run] Writing " << mOutputDigits.size() << " digits. IR rejected:" << mDecoder.getIRRejected() << " TCR rejected:" << mDecoder.getTCRRejected();
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
  }
  if (mDoLumi) {
    uint32_t tfCountsT = 0;
    uint32_t tfCountsV = 0;
    for (auto const& lp : lumiPointsHBF1) {
      tfCountsT += lp.counts;
      tfCountsV += lp.countsFV0;
    }
    // LOG(info) << "Lumi rate:" << tfCounts/(128.*88e-6);
    // FT0
    mHistoryT.push_back(tfCountsT);
    mCountsT += tfCountsT;
    if (mHistoryT.size() <= mNTFToIntegrate) {
      mNHBIntegratedT += lumiPointsHBF1.size();
    } else {
      mCountsT -= mHistoryT.front();
      mHistoryT.pop_front();
    }
    // FV0
    mHistoryV.push_back(tfCountsV);
    mCountsV += tfCountsV;
    if (mHistoryV.size() <= mNTFToIntegrate) {
      mNHBIntegratedV += lumiPointsHBF1.size();
    } else {
      mCountsV -= mHistoryV.front();
      mHistoryV.pop_front();
    }
    //
    if (mNHBIntegratedT || mNHBIntegratedV) {
      mOutputLumiInfo.orbit = lumiPointsHBF1[0].orbit;
    }
    mOutputLumiInfo.counts = mCountsT;
    mOutputLumiInfo.countsFV0 = mCountsV;
    mOutputLumiInfo.nHBFCounted = mNHBIntegratedT;
    mOutputLumiInfo.nHBFCountedFV0 = mNHBIntegratedV;
    if (mVerbose) {
      LOGP(info, "Orbit {}: {}/{} counts T/V in {}/{} HBFs -> lumiT = {:.3e}+-{:.3e} lumiV = {:.3e}+-{:.3e}", mOutputLumiInfo.orbit, mCountsT, mCountsV, mNHBIntegratedT, mNHBIntegratedV, mOutputLumiInfo.getLumi(), mOutputLumiInfo.getLumiError(), mOutputLumiInfo.getLumiFV0(), mOutputLumiInfo.getLumiFV0Error());
    }
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}, mOutputLumiInfo);
  }
}
o2::framework::DataProcessorSpec o2::ctp::reco_workflow::getRawDecoderSpec(bool askDISTSTF, bool digits, bool lumi)
{
  if (!digits && !lumi) {
    throw std::runtime_error("all outputs were disabled");
  }
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, o2::framework::Lifetime::Optional);
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  std::vector<o2::framework::OutputSpec> outputs;
  if (digits) {
    outputs.emplace_back("CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  }
  if (lumi) {
    outputs.emplace_back("CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe);
  }
  return o2::framework::DataProcessorSpec{
    "ctp-raw-decoder",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::ctp::reco_workflow::RawDecoderSpec>(digits, lumi)},
    o2::framework::Options{
      {"ntf-to-average", o2::framework::VariantType::Int, 90, {"Time interval for averaging luminosity in units of TF"}},
      {"use-verbose-mode", o2::framework::VariantType::Bool, false, {"Verbose logging"}}}};
}
