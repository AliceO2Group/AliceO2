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
#include "Framework/WorkflowSpec.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"
#include "CTPWorkflow/RawToDigitConverterSpec.h"

using namespace o2::ctp::reco_workflow;

void RawToDigitConverterSpec::init(framework::InitContext& ctx)
{
}

void RawToDigitConverterSpec::run(framework::ProcessingContext& ctx)
{
  mOutputDigits.clear();
  std::map<o2::InteractionRecord, CTPDigit> digits;
  const gbtword80_t bcidmask = 0xfff;
  gbtword80_t pldmask;
  using InputSpec = o2::framework::InputSpec;
  using ConcreteDataTypeMatcher = o2::framework::ConcreteDataTypeMatcher;
  using Lifetime = o2::framework::Lifetime;
  //mOutputHWErrors.clear();
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, Lifetime::Timeframe}};
  o2::framework::DPLRawParser parser(ctx.inputs(), filter);
  //setUpDummyLink
  auto& inputs = ctx.inputs();
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    std::vector<InputSpec> dummy{InputSpec{"dummy", o2::framework::ConcreteDataMatcher{"CTP", "RAWDATA", 0xDEADBEEF}}};
    for (const auto& ref : o2::framework::InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (dh->payloadSize == 0) {
        LOGP(WARNING, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
        ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
        return;
      }
    }
  }
  //
  uint32_t payloadCTP;
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    auto rdh = it.get_if<o2::header::RAWDataHeader>();
    auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
    auto linkCRU = o2::raw::RDHUtils::getLinkID(rdh); // 0 = IR, 1 = TCR
    if (linkCRU == o2::ctp::CRULinkIDIntRec) {
      payloadCTP = o2::ctp::NIntRecPayload;
    } else if (linkCRU == o2::ctp::CRULinkIDClassRec) {
      payloadCTP = o2::ctp::NClassPayload;
    } else {
      LOG(ERROR) << "Unxpected  CTP CRU link:" << linkCRU;
    }
    pldmask = 0;
    for (uint32_t i = 0; i < payloadCTP; i++) {
      pldmask[12 + i] = 1;
    }
    // TF in 128 bits words
    gsl::span<const uint8_t> payload(it.data(), it.size());
    gbtword80_t gbtWord = 0;
    gbtword80_t remnant = 0;
    uint32_t size_gbt = 0;
    int wordCount = 0;
    std::vector<gbtword80_t> diglets;
    for (auto payloadWord : payload) {
      if (wordCount == 15) {
        wordCount = 0;
      } else if (wordCount > 9) {
        wordCount++;
      } else if (wordCount == 9) {
        wordCount++;
        diglets.clear();
        makeGBTWordInverse(diglets, gbtWord, remnant, size_gbt, payloadCTP);
        // save digit in buffer recs
        for (auto diglet : diglets) {
          gbtword80_t pld = (diglet & pldmask);
          if (pld.count() == 0) {
            continue;
          }
          pld <<= 12;
          CTPDigit digit;
          uint32_t bcid = (diglet & bcidmask).to_ulong();
          o2::InteractionRecord ir;
          ir.orbit = triggerOrbit;
          ir.bc = bcid;
          digit.intRecord = ir;
          if (linkCRU == o2::ctp::CRULinkIDIntRec) {
            if (digits.count(ir) == 1) {
              if (digits[ir].CTPInputMask.count() == 0) {
                digits[ir].setInputMask(pld);
              } else {
                LOG(ERROR) << "Two CTP IRs for same timestamp.";
              }
            } else {
              digit.setInputMask(pld);
              digits[ir] = digit;
            }
          } else if (linkCRU == o2::ctp::CRULinkIDClassRec) {
            if (digits.count(ir) == 1) {
              if (digits[ir].CTPClassMask.count() == 0) {
                digits[ir].setClassMask(pld);
              } else {
                LOG(ERROR) << "Two CTP Class masks for same timestamp";
              }
            } else {
              digit.setClassMask(pld);
              digits[ir] = digit;
            }
          } else {
            LOG(ERROR) << "Unxpected  CTP CRU link:" << linkCRU;
          }
        }
        gbtWord = 0;
      } else {
        gbtWord |= payloadWord >> (wordCount * 8);
        wordCount++;
      }
    }
  }
  for (auto const digmap : digits) {
    mOutputDigits.push_back(digmap.second);
  }

  LOG(INFO) << "[CTPRawToDigitConverter - run] Writing " << mOutputDigits.size() << " digits ...";
  ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
  //ctx.outputs().snapshot(o2::framework::Output{"CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe}, mOutputHWErrors);
}
// Inverse of Digits2Raw::makeGBTWord
void RawToDigitConverterSpec::makeGBTWordInverse(std::vector<gbtword80_t>& diglets, gbtword80_t& GBTWord, gbtword80_t& remnant, uint32_t& size_gbt, uint32_t Npld) const
{
  gbtword80_t diglet = remnant;
  uint32_t i = 0;
  while (i < (NGBT - Npld)) {
    std::bitset<NGBT> masksize = 0;
    for (uint32_t j = 0; j < (Npld - size_gbt); j++) {
      masksize[j] = 1;
    }
    diglet |= (GBTWord & masksize) << (size_gbt);
    diglets.push_back(diglet);
    diglet = 0;
    i += Npld - size_gbt;
    GBTWord = GBTWord >> (Npld - size_gbt);
    size_gbt = 0;
  }
  size_gbt = NGBT - i;
  remnant = GBTWord;
}
o2::framework::DataProcessorSpec o2::ctp::reco_workflow::getRawToDigitConverterSpec(bool askDISTSTF)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, o2::framework::Lifetime::Optional);
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{
    "CTP-RawStreamDecoder",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::ctp::reco_workflow::RawToDigitConverterSpec>()},
    o2::framework::Options{{"result-file", o2::framework::VariantType::String, "/tmp/hmpCTPDecodeResults", {"Base name of the decoding results files."}}}};
}
