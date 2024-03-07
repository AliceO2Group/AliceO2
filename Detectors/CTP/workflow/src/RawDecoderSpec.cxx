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
#include "Framework/InputRecord.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"

using namespace o2::ctp::reco_workflow;

void RawDecoderSpec::init(framework::InitContext& ctx)
{
  bool decodeinps = ctx.options().get<bool>("ctpinputs-decoding");
  mDecoder.setDecodeInps(decodeinps);
  mNTFToIntegrate = ctx.options().get<int>("ntf-to-average");
  mVerbose = ctx.options().get<bool>("use-verbose-mode");
  int maxerrors = ctx.options().get<int>("print-errors-num");
  mDecoder.setVerbose(mVerbose);
  mDecoder.setDoLumi(mDoLumi);
  mDecoder.setDoDigits(mDoDigits);
  mDecoder.setMAXErrors(maxerrors);
  std::string lumiinp1 = ctx.options().get<std::string>("lumi-inp1");
  std::string lumiinp2 = ctx.options().get<std::string>("lumi-inp2");
  int inp1 = mDecoder.setLumiInp(1, lumiinp1);
  int inp2 = mDecoder.setLumiInp(2, lumiinp2);
  mOutputLumiInfo.inp1 = inp1;
  mOutputLumiInfo.inp2 = inp2;
  mMaxOutputSize = ctx.options().get<int>("max-output-size");
  LOG(info) << "CTP reco init done. Inputs decoding here:" << decodeinps << " DoLumi:" << mDoLumi << " DoDigits:" << mDoDigits << " NTF:" << mNTFToIntegrate << " Lumi inputs:" << lumiinp1 << ":" << inp1 << " " << lumiinp2 << ":" << inp2 << " Max errors:" << maxerrors << " Max output size:" << mMaxOutputSize;
  // mOutputLumiInfo.printInputs();
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
  std::cout << "# of IR errors:" << mDecoder.getErrorIR() << " TCR errors:" << mDecoder.getErrorTCR() << std::endl;
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
      ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0}, this->mOutputDigits);
    }
    if (this->mDoLumi) {
      ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0}, this->mOutputLumiInfo);
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
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, Lifetime::Timeframe}};
  int ret = mDecoder.decodeRaw(inputs, filter, mOutputDigits, lumiPointsHBF1);
  if (ret == 1) {
    dummyOutput();
    return;
  }
  if (mDoDigits) {
    LOG(info) << "[CTPRawToDigitConverter - run] Writing " << mOutputDigits.size() << " digits. IR rejected:" << mDecoder.getIRRejected() << " TCR rejected:" << mDecoder.getTCRRejected();
    if( (mMaxOutputSize > 0) && (mOutputDigits.size() > mMaxOutputSize)) {
      LOG(error) << "CTP raw output size: " << mOutputDigits.size();
      mOutputDigits.clear();
    }
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0}, mOutputDigits);
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
      mOutputLumiInfo.printInputs();
      LOGP(info, "Orbit {}: {}/{} counts inp1/inp2 in {}/{} HBFs -> lumi_inp1 = {:.3e}+-{:.3e} lumi_inp2 = {:.3e}+-{:.3e}", mOutputLumiInfo.orbit, mCountsT, mCountsV, mNHBIntegratedT, mNHBIntegratedV, mOutputLumiInfo.getLumi(), mOutputLumiInfo.getLumiError(), mOutputLumiInfo.getLumiFV0(), mOutputLumiInfo.getLumiFV0Error());
    }
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0}, mOutputLumiInfo);
  }
}
o2::framework::DataProcessorSpec o2::ctp::reco_workflow::getRawDecoderSpec(bool askDISTSTF, bool digits, bool lumi)
{
  if (!digits && !lumi) {
    throw std::runtime_error("all outputs were disabled");
  }
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, o2::framework::Lifetime::Timeframe);
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
      {"print-errors-num", o2::framework::VariantType::Int, 3, {"Max number of errors to print"}},
      {"lumi-inp1", o2::framework::VariantType::String, "TVX", {"The first input used for online lumi. Name in capital."}},
      {"lumi-inp2", o2::framework::VariantType::String, "VBA", {"The second input used for online lumi. Name in capital."}},
      {"use-verbose-mode", o2::framework::VariantType::Bool, false, {"Verbose logging"}},
      {"max-output-size", o2::framework::VariantType::Int,0,{"Do not send output if bigger than max size, 0 - do not check"}},
      {"ctpinputs-decoding", o2::framework::VariantType::Bool, false, {"Inputs alignment: true - raw decoder - has to be compatible with CTF decoder: allowed options: 10,01,00"}}}};
}
