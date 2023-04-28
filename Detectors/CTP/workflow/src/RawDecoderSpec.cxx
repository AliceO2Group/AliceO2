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
#include "DPLUtils/DPLRawParser.h"
#include "CTPWorkflow/RawDecoderSpec.h"
#include "CommonUtils/VerbosityConfig.h"

using namespace o2::ctp::reco_workflow;

void RawDecoderSpec::init(framework::InitContext& ctx)
{
  mNTFToIntegrate = ctx.options().get<int>("ntf-to-average");
  mVerbose = ctx.options().get<bool>("use-verbose-mode");
  LOG(info) << "CTP reco init done";
}
void RawDecoderSpec::endOfStream(framework::EndOfStreamContext& ec)
{
  std::sort(mTFOrbits.begin(), mTFOrbits.end());
  size_t l = mTFOrbits.size();
  uint32_t o0 = 0;
  if (l) {
    o0 = mTFOrbits[0];
  }
  int nmiss = 0;
  int nprt = 0;
  std::cout << "Missing orbits:";
  for (int i = 1; i < l; i++) {
    if ((mTFOrbits[i] - o0) > 0x20) {
      if (nprt < 20) {
        std::cout << " " << o0 << "-" << mTFOrbits[i];
      }
      nmiss += (mTFOrbits[i] - o0) / 0x20;
      nprt++;
    }
    o0 = mTFOrbits[i];
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
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{"CTP", "RAWDATA"}, Lifetime::Timeframe}};
  o2::framework::DPLRawParser parser(ctx.inputs(), filter);
  std::vector<LumiInfo> lumiPointsHBF1;
  uint64_t countsMBT = 0;
  uint64_t countsMBV = 0;
  uint32_t payloadCTP;
  gbtword80_t remnant = 0;
  uint32_t size_gbt = 0;
  mTFOrbit = 0;
  uint32_t orbit0 = 0;
  for (auto it = parser.begin(); it != parser.end(); ++it) {
    const o2::header::RDHAny* rdh = nullptr;
    try {
      rdh = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      mPadding = (o2::raw::RDHUtils::getDataFormat(rdh) == 0);
    } catch (std::exception& e) {
      LOG(error) << "Failed to extract RDH, abandoning TF sending dummy output, exception was: " << e.what();
      dummyOutput();
      return;
    }
    // auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
    uint32_t stopBit = o2::raw::RDHUtils::getStop(rdh);
    uint32_t packetCounter = o2::raw::RDHUtils::getPageCounter(rdh);
    uint32_t version = o2::raw::RDHUtils::getVersion(rdh);
    uint32_t rdhOrbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdh);
    uint32_t triggerType = o2::raw::RDHUtils::getTriggerType(rdh);
    // std::cout << "diff orbits:" << triggerOrbit - rdhOrbit << std::endl;
    bool tf = (triggerType & TF_TRIGGERTYPE_MASK) && (packetCounter == 0);
    bool hb = (triggerType & HB_TRIGGERTYPE_MASK) && (packetCounter == 0);
    if (tf) {
      mTFOrbit = rdhOrbit;
      // std::cout << "tforbit==================>" << mTFOrbit << " " << std::hex << mTFOrbit << std::endl;
      mTFOrbits.push_back(mTFOrbit);
    }
    static bool prt = true;
    if (prt) {
      LOG(info) << "RDH version:" << version << " Padding:" << mPadding;
      prt = false;
    }
    auto feeID = o2::raw::RDHUtils::getFEEID(rdh); // 0 = IR, 1 = TCR
    auto linkCRU = (feeID & 0xf00) >> 8;
    if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
      payloadCTP = o2::ctp::NIntRecPayload;
    } else if (linkCRU == o2::ctp::GBTLinkIDClassRec) {
      payloadCTP = o2::ctp::NClassPayload;
      if (!mDoDigits) { // Do not do TCR if only lumi
        continue;
      }
    } else {
      LOG(error) << "Unxpected  CTP CRU link:" << linkCRU;
    }
    LOG(debug) << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << rdhOrbit << " triggerType:" << triggerType;
    // LOG(info) << "remnant :" << remnant.count();
    gbtword80_t pldmask = 0;
    for (uint32_t i = 0; i < payloadCTP; i++) {
      pldmask[12 + i] = 1;
    }
    // std::cout << (orbit0 != rdhOrbit) << " comp " << (mTFOrbit==rdhOrbit) << std::endl;
    // if(orbit0 != rdhOrbit) {
    if (hb) {
      if (mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) { // create lumi per HB
        lumiPointsHBF1.emplace_back(LumiInfo{rdhOrbit, 0, 0, countsMBT, countsMBV});
        countsMBT = 0;
        countsMBV = 0;
      }
      remnant = 0;
      size_gbt = 0;
      orbit0 = rdhOrbit;
      // std::cout << "orbit0============>" << std::dec << orbit0 << " " << std::hex << orbit0 << std::endl;
    }
    // Create 80 bit words
    gsl::span<const uint8_t> payload(it.data(), it.size());
    gbtword80_t gbtWord80;
    gbtWord80.set();
    int wordCount = 0;
    int wordSize = 10;
    std::vector<gbtword80_t> gbtwords80;
    // mPadding = 0;
    if (mPadding == 1) {
      wordSize = 16;
    }
    /* if (payload.size()) {
      //LOG(info) << "payload size:" << payload.size();
      // LOG(info) << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << triggerOrbit << " stopbit:" << stopBit << " packet:" << packetCounter;
      // LOGP(info, "RDH FEEid: {} CRU link: {}, Orbit: {}", feeID, linkCRU, triggerOrbit);
      std::cout << std::hex << "RDH FEEid: " << feeID << " CTP CRU link:" << linkCRU << " Orbit:" << rdhOrbit << std::endl;
    } */
    gbtword80_t bcmask = std::bitset<80>("111111111111");
    for (auto payloadWord : payload) {
      int wc = wordCount % wordSize;
      // LOG(info) << wordCount << ":" << wc << " payload:" << int(payloadWord);
      if ((wc == 0) && (wordCount != 0)) {
        if (gbtWord80.count() != 80) {
          gbtwords80.push_back(gbtWord80);
          /* uint64_t bcid = (gbtWord80 & bcmask).to_ullong();
          if (bcid < 279)
            bcid += 3564 - 279;
          else
            bcid += -279;
          std::string ss = fmt::format("{:x}", bcid);
          LOG(info) << "w80:" << gbtWord80 << " " << ss;
          // LOGP(info,"w80: {} bcid:{%x}", gbtWord80,bcid); */
        }
        gbtWord80.set();
      }
      if (wc < 10) {
        for (int i = 0; i < 8; i++) {
          gbtWord80[wc * 8 + i] = bool(int(payloadWord) & (1 << i));
        }
      }
      wordCount++;
    }
    if ((gbtWord80.count() != 80) && (gbtWord80.count() > 0)) {
      gbtwords80.push_back(gbtWord80);
      /*uint64_t bcid = (gbtWord80 & bcmask).to_ullong();
      if (bcid < 279)
        bcid += 3564 - 279;
      else
        bcid += -279;
      std::string ss = fmt::format("{:x}", bcid);
      LOG(info) << "w80l:" << gbtWord80 << " " << ss;  */
    }
    // decode 80 bits payload
    for (auto word : gbtwords80) {
      std::vector<gbtword80_t> diglets;
      gbtword80_t gbtWord = word;
      makeGBTWordInverse(diglets, gbtWord, remnant, size_gbt, payloadCTP);
      for (auto diglet : diglets) {
        if (mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) {
          gbtword80_t pld = (diglet >> 12) & mTVXMask;
          if (pld.count() != 0) {
            countsMBT++;
          }
          pld = (diglet >> 12) & mVBAMask;
          if (pld.count() != 0) {
            countsMBV++;
          }
        }
        if (!mDoDigits) {
          continue;
        }
        LOG(debug) << "diglet:" << diglet << " " << (diglet & bcmask).to_ullong();
        addCTPDigit(linkCRU, rdhOrbit, diglet, pldmask, digits);
      }
    }
    // if ((remnant.count() > 0) && stopBit) {
    if (remnant.count() > 0) {
      if (mDoLumi && payloadCTP == o2::ctp::NIntRecPayload) {
        gbtword80_t pld = (remnant >> 12) & mTVXMask;
        if (pld.count() != 0) {
          countsMBT++;
        }
        pld = (remnant >> 12) & mVBAMask;
        if (pld.count() != 0) {
          countsMBV++;
        }
      }
      if (!mDoDigits) {
        continue;
      }
      addCTPDigit(linkCRU, rdhOrbit, remnant, pldmask, digits);
      LOG(debug) << "diglet:" << remnant << " " << (remnant & bcmask).to_ullong();
      remnant = 0;
    }
  }
  if (mDoDigits) {
    for (auto const digmap : digits) {
      mOutputDigits.push_back(digmap.second);
    }
    LOG(info) << "[CTPRawToDigitConverter - run] Writing " << mOutputDigits.size() << " digits. IR rejected:" << mIRRejected << " TCR rejected:" << mTCRRejected;
    ctx.outputs().snapshot(o2::framework::Output{"CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
  }
  if (mDoLumi) {
    lumiPointsHBF1.emplace_back(LumiInfo{orbit0, 0, 0, countsMBT, countsMBV});
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

// Inverse of Digits2Raw::makeGBTWord
void RawDecoderSpec::makeGBTWordInverse(std::vector<gbtword80_t>& diglets, gbtword80_t& GBTWord, gbtword80_t& remnant, uint32_t& size_gbt, uint32_t Npld)
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
int RawDecoderSpec::addCTPDigit(uint32_t linkCRU, uint32_t orbit, gbtword80_t& diglet, gbtword80_t& pldmask, std::map<o2::InteractionRecord, CTPDigit>& digits)
{
  gbtword80_t pld = (diglet & pldmask);
  if (pld.count() == 0) {
    return 0;
  }
  pld >>= 12;
  CTPDigit digit;
  const gbtword80_t bcidmask = 0xfff;
  uint16_t bcid = (diglet & bcidmask).to_ulong();
  LOG(debug) << bcid << "    pld:" << pld;
  o2::InteractionRecord ir = {bcid, orbit};
  int32_t BCShiftCorrection = o2::ctp::TriggerOffsetsParam::Instance().customOffset[o2::detectors::DetID::CTP];
  if (linkCRU == o2::ctp::GBTLinkIDIntRec) {
    LOG(debug) << "InputMaskCount:" << digits[ir].CTPInputMask.count();
    LOG(debug) << "ir ir ori:" << ir;
    // if ((int32_t)ir.bc < BCShiftCorrection) {
    if ((ir.orbit <= mTFOrbit) && ((int32_t)ir.bc < BCShiftCorrection)) {
      // LOG(warning) << "Loosing ir:" << ir;
      mIRRejected++;
      return 0;
    }
    ir -= BCShiftCorrection;
    LOG(debug) << "ir ir corrected:" << ir;
    digit.intRecord = ir;
    if (digits.count(ir) == 0) {
      digit.setInputMask(pld);
      digits[ir] = digit;
      LOG(debug) << bcid << " inputs case 0 bcid orbit " << orbit << " pld:" << pld;
    } else if (digits.count(ir) == 1) {
      if (digits[ir].CTPInputMask.count() == 0) {
        digits[ir].setInputMask(pld);
        LOG(debug) << bcid << " inputs bcid vase 1 orbit " << orbit << " pld:" << pld;
      } else {
        LOG(error) << "Two CTP IRs with the same timestamp:" << ir.bc << " " << ir.orbit;
      }
    } else {
      LOG(error) << "Two digits with the same rimestamp:" << ir.bc << " " << ir.orbit;
    }
  } else if (linkCRU == o2::ctp::GBTLinkIDClassRec) {
    int32_t offset = BCShiftCorrection + o2::ctp::TriggerOffsetsParam::Instance().LM_L0 + o2::ctp::TriggerOffsetsParam::Instance().L0_L1 - 1;
    LOG(debug) << "tcr ir ori:" << ir;
    // if ((int32_t)ir.bc < offset) {
    if ((ir.orbit <= mTFOrbit) && ((int32_t)ir.bc < offset)) {
      // if (0) {
      LOG(debug) << "Loosing tclass:" << ir;
      mTCRRejected++;
      return 0;
    }
    ir -= offset;
    LOG(debug) << "tcr ir corrected:" << ir;
    digit.intRecord = ir;
    if (digits.count(ir) == 0) {
      digit.setClassMask(pld);
      digits[ir] = digit;
      LOG(debug) << bcid << " class bcid case 0 orbit " << orbit << " pld:" << pld;
    } else if (digits.count(ir) == 1) {
      if (digits[ir].CTPClassMask.count() == 0) {
        digits[ir].setClassMask(pld);
        LOG(debug) << bcid << " class bcid case 1 orbit " << orbit << " pld:" << pld;
      } else {
        LOG(error) << "Two CTP Class masks for same timestamp";
      }
    } else {
    }
  } else {
    LOG(error) << "Unxpected  CTP CRU link:" << linkCRU;
  }
  return 0;
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
