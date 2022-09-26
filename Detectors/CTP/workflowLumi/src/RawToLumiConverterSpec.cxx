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
#include "Framework/ConfigParamRegistry.h"
#include "CTPWorkflowLumi/RawToLumiConverterSpec.h"
#include "CTPWorkflow/RawToDigitConverterSpec.h"
#include "CommonUtils/VerbosityConfig.h"

using namespace o2::ctp::lumi_workflow;

void RawToLumiConverterSpec::init(framework::InitContext& ctx)
{
  mHBFsToAverage = ctx.options().get<int>("nHBF-to-IRaverage");
}

void RawToLumiConverterSpec::run(framework::ProcessingContext& ctx)
{
  const gbtword80_t bcidmask = 0xfff;
  using InputSpec = o2::framework::InputSpec;
  using ConcreteDataTypeMatcher = o2::framework::ConcreteDataTypeMatcher;
  using Lifetime = o2::framework::Lifetime;
  // mOutputHWErrors.clear();
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, Lifetime::Timeframe}};
  o2::framework::DPLRawParser parser(ctx.inputs(), filter);
  // setUpDummyLink
  auto& inputs = ctx.inputs();
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
        ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}, mOutputLumiPoints);
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }
  //
  std::vector<lumiPoint> lumiPointsHBF1;
  float_t countsMB = 0;
  uint32_t payloadCTP;
  uint32_t orbit0 = 0;
  bool first = true;
  gbtword80_t remnant = 0;
  uint32_t size_gbt = 0;
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    auto rdh = it.get_if<o2::header::RAWDataHeader>();
    auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);

    auto feeID = o2::raw::RDHUtils::getFEEID(rdh); // 0 = IR, 1 = TCR
    auto linkCRU = (feeID & 0xf00) >> 8;
    if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
      payloadCTP = o2::ctp::NIntRecPayload;
    } else {
      continue;
    }
    if (first) {
      orbit0 = triggerOrbit;
      first = false;
    }
    LOG(debug) << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << triggerOrbit;
    // TF in 128 bits words
    gsl::span<const uint8_t> payload(it.data(), it.size());
    gbtword80_t gbtWord = 0;
    int wordCount = 0;
    std::vector<gbtword80_t> diglets;
    if (orbit0 != triggerOrbit) {
      // create lumi per HB
      lumiPoint lp;
      lp.counts = countsMB;
      lp.ir.orbit = triggerOrbit;
      lumiPointsHBF1.push_back(lp);
      //LOG(info) << "Orbit:" << triggerOrbit << " tvx count:" << countsMB;
      countsMB = 0;
      //
      remnant = 0;
      size_gbt = 0;
      orbit0 = triggerOrbit;
    }
    for (auto payloadWord : payload) {
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
            countsMB++;
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
  //LOG(info) << "lumiPoints size:" << lumiPointsHBF1.size();
  mOutputLumiPoints.clear();
  float_t countHBFn = 0;
  for(int ihb = 0; ihb < lumiPointsHBF1.size(); ihb++) {
    countHBFn += lumiPointsHBF1[ihb].counts;
    lumiPoint lp;
    if(ihb < mHBFsToAverage) {
      lp.nHBFs = ihb+1;
      lp.ir = lumiPointsHBF1[ihb].ir;
    } else {
      countHBFn -= lumiPointsHBF1[ihb - mHBFsToAverage].counts;
      lp.nHBFs = mHBFsToAverage;
      lp.ir = lumiPointsHBF1[ihb - mHBFsToAverage + 1].ir;
    }
    lp.counts = countHBFn;
    //LOG(info) << "lp orbit:" << lp.ir.orbit << " lp.countes:" << lp.counts << " T:" << lp.nHBFs;
    mOutputLumiPoints.push_back(lp);
  }
  LOG(info) << "[CTPRawToLumiConverter - run] Writing " << mOutputLumiPoints.size() << " lumiPoints ...";
  ctx.outputs().snapshot(o2::framework::Output{"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}, mOutputLumiPoints);
}
o2::framework::DataProcessorSpec o2::ctp::lumi_workflow::getRawToLumiConverterSpec(bool askDISTSTF)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"CTP", "RAWDATA"},  o2::framework::Lifetime::Optional);

  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{
    "CTP-RawStreamDecoder",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::ctp::lumi_workflow::RawToLumiConverterSpec>()},
    o2::framework::Options{{"nHBF-to-IRaverage", o2::framework::VariantType::Int, 1, {"Time interval for averaging IR rate in units of HB"}}}
  };
}
