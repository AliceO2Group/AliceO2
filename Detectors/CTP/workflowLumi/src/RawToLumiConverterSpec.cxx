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
#include "FairLogger.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "Framework/WorkflowSpec.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"
#include "CTPWorkflowLumi/RawToLumiConverterSpec.h"
#include "CTPWorkflow/RawToDigitConverterSpec.h"
#include "CommonUtils/VerbosityConfig.h"

using namespace o2::ctp::lumi_workflow;

void RawToLumiConverterSpec::init(framework::InitContext& ctx)
{
}

void RawToLumiConverterSpec::run(framework::ProcessingContext& ctx)
{
  mOutputLumiPoints.clear();
  const gbtword80_t bcidmask = 0xfff;
  using InputSpec = o2::framework::InputSpec;
  using ConcreteDataTypeMatcher = o2::framework::ConcreteDataTypeMatcher;
  using Lifetime = o2::framework::Lifetime;
  auto& inputs = ctx.inputs();
  auto ref = inputs.get("TF", o2::ctp::GBTLinkIDIntRec);
  auto rdh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
  auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
  float_t counts = 0;
  uint32_t payloadCTP = o2::ctp::NIntRecPayload;
  uint32_t orbit0 = 0;
  bool first = true;
  gbtword80_t remnant = 0;
  uint32_t size_gbt = 0;
  // for (auto it = parser.begin(); it != parser.end(); ++it)
  {
    // auto rdh = it.get_if<o2::header::RAWDataHeader>();
    auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
    LOG(info) << "====>" << payloadSize << " " << triggerOrbit;
    return;
    if (first) {
      orbit0 = triggerOrbit;
      first = false;
    }
    // TF in 128 bits words
    auto payload = inputs.get<const uint8_t*>(ref);
    // gsl::span<const uint8_t> payload(pld.data(), pld.size());
    gbtword80_t gbtWord = 0;
    int wordCount = 0;
    std::vector<gbtword80_t> diglets;
    if (orbit0 != triggerOrbit) {
      // create lumi per HB
      lumiPoint lp;
      lp.counts = counts;
      lp.ir.orbit = triggerOrbit;
      mOutputLumiPoints.push_back(lp);
      //
      remnant = 0;
      size_gbt = 0;
      orbit0 = triggerOrbit;
      counts = 0;
    }
    // for (auto it = parser.begin(); it != parser.end(); ++it)
    // for (auto it = payload.begin(); it != payload.end(); ++it)
    int payloadWord = 0;
    {
      // LOG(info) << wordCount << " payload:" <<  int(payloadWord);
      if (wordCount == 15) {
        wordCount = 0;
      } else if (wordCount > 9) {
        wordCount++;
      } else if (wordCount == 9) {
        for (int i = 0; i < 8; i++) {
          gbtWord[wordCount * 8 + i] = bool(int(payloadWord) & (1 << i));
        }
        wordCount++;
        diglets.clear();
        // LOG(info) << " gbtword:" << gbtWord;
        o2::ctp::reco_workflow::RawToDigitConverterSpec::makeGBTWordInverse(diglets, gbtWord, remnant, size_gbt, payloadCTP);
        // count tvx
        for (auto diglet : diglets) {
          // LOG(info) << " diglet:" << diglet;
          gbtword80_t pld = (diglet & mTVXMask);
          if (pld.count() != 0) {
            counts++;
          }
        }
        gbtWord = 0;
      } else {
        // std::cout << "wordCount:" << wordCount << std::endl;
        for (int i = 0; i < 8; i++) {
          gbtWord[wordCount * 8 + i] = bool(int(payloadWord) & (1 << i));
        }
        wordCount++;
      }
    }
  }
  LOG(info) << "[CTPRawToLumiConverter - run] Writing " << mOutputLumiPoints.size() << " lumiPoints ...";
  ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}, mOutputLumiPoints);
}
o2::framework::DataProcessorSpec o2::ctp::lumi_workflow::getRawToLumiConverterSpec(bool askDISTSTF)
{
  std::vector<o2::framework::InputSpec> inputs;
  // ok ?
  inputs.emplace_back("TF", "CTP", "RAWDATA", o2::ctp::GBTLinkIDIntRec, o2::framework::Lifetime::Optional);

  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{
    "CTP-RawStreamDecoder",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::ctp::lumi_workflow::RawToLumiConverterSpec>()}
    // o2::framework::Options{{"result-file", o2::framework::VariantType::String, "/tmp/lumiCTPDecodeResults", {"Base name of the decoding results files."}}}
  };
}
