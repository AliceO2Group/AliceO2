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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "DataFormatsCPV/CPVBlockHeader.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "DetectorsRaw/RDHUtils.h"
#include "CPVReconstruction/RawDecoder.h"
#include "CPVWorkflow/RawToDigitConverterSpec.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "CPVBase/Geometry.h"
#include "CommonUtils/VerbosityConfig.h"
#include "CommonUtils/NameConf.h"

using namespace o2::cpv::reco_workflow;
using ConcreteDataMatcher = o2::framework::ConcreteDataMatcher;
using Lifetime = o2::framework::Lifetime;

void RawToDigitConverterSpec::init(framework::InitContext& ctx)
{
  mStartTime = std::chrono::system_clock::now();
  mDecoderErrorsPerMinute = 0;
  mIsMuteDecoderErrors = false;

  LOG(debug) << "Initializing RawToDigitConverterSpec...";
  // Pedestal flag true/false
  LOG(info) << "Pedestal run: " << (mIsPedestalData ? "YES" : "NO");
  if (mIsPedestalData) { // no calibration for pedestal runs needed
    mIsUsingGainCalibration = false;
    mIsUsingBadMap = false;
    LOG(info) << "CCDB is not used. Task configuration is done.";
    return; // all other flags are irrelevant in this case
  }

  // is use-gain-calibration flag setted?
  LOG(info) << "Gain calibration is " << (mIsUsingGainCalibration ? "ON" : "OFF");

  // is use-bad-channel-map flag setted?
  LOG(info) << "Bad channel rejection is " << (mIsUsingBadMap ? "ON" : "OFF");

  LOG(info) << "Task configuration is done.";
}

void RawToDigitConverterSpec::finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == framework::ConcreteDataMatcher("CTP", "Trig_Offset", 0)) {
    LOG(info) << "RawToDigitConverterSpec::finaliseCCDB() : CTP/Config/TriggerOffsets updated.";
    const auto& par = o2::ctp::TriggerOffsetsParam::Instance();
    par.printKeyValues();
    return;
  }
}

void RawToDigitConverterSpec::updateTimeDependentParams(framework::ProcessingContext& ctx)
{
  static bool updateOnlyOnce = false;
  if (!updateOnlyOnce) {
    ctx.inputs().get<o2::ctp::TriggerOffsetsParam*>("trigoffset");

    // std::decay_t<decltype(ctx.inputs().get<o2::cpv::Pedestals*>("peds"))> pedPtr{};
    // std::decay_t<decltype(ctx.inputs().get<o2::cpv::BadChannelMap*>("badmap"))> badMapPtr{};
    // std::decay_t<decltype(ctx.inputs().get<o2::cpv::CalibParams*>("gains"))> gainsPtr{};

    if (!mIsPedestalData) {
      auto pedPtr = ctx.inputs().get<o2::cpv::Pedestals*>("peds");
      mPedestals = pedPtr.get();
    }

    if (mIsUsingBadMap) {
      auto badMapPtr = ctx.inputs().get<o2::cpv::BadChannelMap*>("badmap");
      mBadMap = badMapPtr.get();
    }

    if (mIsUsingGainCalibration) {
      auto gainsPtr = ctx.inputs().get<o2::cpv::CalibParams*>("gains");
      mGains = gainsPtr.get();
    }
    updateOnlyOnce = true;
  }
}

void RawToDigitConverterSpec::run(framework::ProcessingContext& ctx)
{
  updateTimeDependentParams(ctx);
  // check timers if we need mute/unmute error reporting
  auto now = std::chrono::system_clock::now();
  if (mIsMuteDecoderErrors) { // check if 10-minutes muting period passed
    if (((now - mTimeWhenMuted) / std::chrono::minutes(1)) >= 10) {
      mIsMuteDecoderErrors = false; // unmute
      if (mDecoderErrorsCounterWhenMuted) {
        LOG(error) << "RawToDigitConverterSpec::run() : " << mDecoderErrorsCounterWhenMuted << " errors happened while it was muted ((";
      }
      mDecoderErrorsCounterWhenMuted = 0;
    }
  }
  if (((now - mStartTime) / std::chrono::minutes(1)) > mMinutesPassed) {
    mMinutesPassed = (now - mStartTime) / std::chrono::minutes(1);
    LOG(debug) << "minutes passed: " << mMinutesPassed;
    mDecoderErrorsPerMinute = 0;
  }

  // Cache digits from bunch crossings as the component reads timeframes from many links consecutively
  std::map<o2::InteractionRecord, std::shared_ptr<std::vector<o2::cpv::Digit>>> digitBuffer; // Internal digit buffer
  mOutputHWErrors.clear();

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
  std::vector<o2::framework::InputSpec> dummy{o2::framework::InputSpec{"dummy", o2::framework::ConcreteDataMatcher{"CPV", o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
  for (const auto& ref : framework::InputRecordWalker(ctx.inputs(), dummy)) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
    if (payloadSize == 0) { // send empty output
      auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
      if (++contDeadBeef <= maxWarn) {
        LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
             contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
      }
      mOutputDigits.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
      mOutputTriggerRecords.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggerRecords);
      mOutputHWErrors.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe}, mOutputHWErrors);
      return; // empty TF, nothing to process
    }
  }
  contDeadBeef = 0; // if good data, reset the counter

  bool skipTF = false; // skip TF due to fatal error?

  std::vector<o2::framework::InputSpec> rawFilter{
    {"RAWDATA", o2::framework::ConcreteDataTypeMatcher{"CPV", "RAWDATA"}, o2::framework::Lifetime::Timeframe},
  };
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs(), rawFilter)) {
    o2::cpv::RawReaderMemory rawreader(o2::framework::DataRefUtils::as<const char>(rawData));
    // loop over all the DMA pages
    while (rawreader.hasNext()) {
      try {
        rawreader.next();
      } catch (RawErrorType_t e) {
        if (!mIsMuteDecoderErrors) {
          LOG(error) << "Raw decoding error " << (int)e;
        }
        // add error list
        // RawErrorType_t is defined in O2/Detectors/CPV/reconstruction/include/CPVReconstruction/RawReaderMemory.h
        // RawDecoderError(short c, short d, short g, short p, RawErrorType_t e)
        mOutputHWErrors.emplace_back(-1, 0, 0, 0, e); // Put general errors to non-existing ccId -1
        // if problem in header, abandon this page
        if (e == RawErrorType_t::kRDH_DECODING) { // fatal error -> skip whole TF
          LOG(error) << "RDH decoding error. Skipping this TF";
          skipTF = true;
          break;
        }
        if (e == RawErrorType_t::kPAGE_NOTFOUND ||       // nothing left to read -> skip to next HBF
            e == RawErrorType_t::kNOT_CPV_RDH ||         // not cpv rdh -> skip to next HBF
            e == RawErrorType_t::kOFFSET_TO_NEXT_IS_0) { // offset to next package is 0 -> do not know how to read next -> skip to next HBF
          break;
        }
        // if problem in payload, try to continue
        continue;
      }
      auto& rdh = rawreader.getRawHeader();
      auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
      auto mod = o2::raw::RDHUtils::getLinkID(rdh) + 2; // link=0,1,2 -> mod=2,3,4
      // for now all modules are written to one LinkID
      if (mod > o2::cpv::Geometry::kNMod || mod < 2) { // only 3 correct modules:2,3,4
        if (!mIsMuteDecoderErrors) {
          LOG(error) << "RDH linkId corresponds to module " << mod << " which does not exist";
        }
        mOutputHWErrors.emplace_back(-1, mod, 0, 0, kRDH_INVALID); // Add non-existing modules to non-existing ccId -1 and dilogic = mod
        continue;
      }
      o2::cpv::RawDecoder decoder(rawreader);
      if (mIsMuteDecoderErrors) {
        decoder.muteErrors();
      }
      RawErrorType_t err = decoder.decode();
      int decoderErrors = 0;
      for (auto errs : decoder.getErrors()) {
        if (errs.ccId == -1) { // error related to wrong data format
          decoderErrors++;
        }
      }
      mDecoderErrorsPerMinute += decoderErrors;
      // LOG(debug) << "RawDecoder found " << decoderErrors << " raw format errors";
      // LOG(debug) << "Now I have " << mDecoderErrorsPerMinute << " errors for current minute";
      if (mIsMuteDecoderErrors) {
        mDecoderErrorsCounterWhenMuted += decoder.getErrors().size();
      } else {
        if (mDecoderErrorsPerMinute > 10) { // mute error reporting for 10 minutes
          LOG(warning) << "> 10 raw decoder error messages per minute, muting it for 10 minutes";
          mIsMuteDecoderErrors = true;
          mTimeWhenMuted = std::chrono::system_clock::now();
        }
      }

      if (!(err == kOK || err == kOK_NO_PAYLOAD)) {
        // TODO handle severe errors
        // TODO: probably careful conversion of decoder errors to Fitter errors?
        mOutputHWErrors.emplace_back(-1, mod, 0, 0, err); // assign general RDH errors to non-existing ccId -1 and dilogic = mod
      }

      std::shared_ptr<std::vector<o2::cpv::Digit>> currentDigitContainer;
      auto digilets = decoder.getDigits();
      if (digilets.empty()) { // no digits -> continue to next pages
        continue;
      }
      o2::InteractionRecord currentIR(0, triggerOrbit); //(bc, orbit)
      // Loop over all the BCs
      for (auto itBCRecords : decoder.getBCRecords()) {
        currentIR.bc = itBCRecords.bc;
        for (unsigned int iDig = itBCRecords.firstDigit; iDig <= itBCRecords.lastDigit; iDig++) {
          auto adch = digilets[iDig];
          auto found = digitBuffer.find(currentIR);
          if (found == digitBuffer.end()) {
            currentDigitContainer = std::make_shared<std::vector<o2::cpv::Digit>>();
            digitBuffer[currentIR] = currentDigitContainer;
          } else {
            currentDigitContainer = found->second;
          }

          AddressCharge ac = {adch};
          unsigned short absId = ac.Address;
          // if we deal with non-pedestal data?
          if (!mIsPedestalData) { // not a pedestal data
            // test bad map
            if (mIsUsingBadMap) {
              if (!mBadMap->isChannelGood(absId)) {
                continue; // skip bad channel
              }
            }
            float amp = 0;
            if (mIsUsingGainCalibration) { // calibrate amplitude
              amp = mGains->getGain(absId) * (ac.Charge - mPedestals->getPedestal(absId));
            } else { // no gain calibration needed
              amp = ac.Charge - mPedestals->getPedestal(absId);
            }
            if (amp > 0) { // emplace new digit
              currentDigitContainer->emplace_back(absId, amp, -1);
            }
          } else { // pedestal data, no calibration needed.
            currentDigitContainer->emplace_back(absId, (float)ac.Charge, -1);
          }
        }
      }
      // Check and send list of hwErrors
      for (auto& er : decoder.getErrors()) {
        mOutputHWErrors.push_back(er);
      }
    }
    // if corrupted raw data is present in this TF -> skip it
    if (skipTF) {
      // Send no digits
      mOutputDigits.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
      mOutputTriggerRecords.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggerRecords);
      // Send errors
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe}, mOutputHWErrors);
      return;
    }
  }

  // Loop over BCs, sort digits with increasing digit ID and write to output containers
  mOutputDigits.clear();
  mOutputTriggerRecords.clear();
  const auto tfOrbitFirst = ctx.services().get<o2::framework::TimingInfo>().firstTForbit;
  const auto& ctpOffsets = o2::ctp::TriggerOffsetsParam::Instance();
  for (auto [bc, digits] : digitBuffer) {
    if (bc.differenceInBC({0, tfOrbitFirst}) < ctpOffsets.LM_L0) {
      continue; // discard this BC as it is actually out of
    }
    int prevDigitSize = mOutputDigits.size();
    if (digits->size()) {
      // Sort digits according to digit ID
      std::sort(digits->begin(), digits->end(), [](o2::cpv::Digit& lhs, o2::cpv::Digit& rhs) { return lhs.getAbsId() < rhs.getAbsId(); });

      for (auto digit : *digits) {
        mOutputDigits.push_back(digit);
      }
    }
    // subtract ctp L0-LM offset here
    mOutputTriggerRecords.emplace_back(bc - ctpOffsets.LM_L0, prevDigitSize, mOutputDigits.size() - prevDigitSize);
  }
  digitBuffer.clear();

  LOG(info) << "[CPVRawToDigitConverter - run] Sending " << mOutputDigits.size() << " digits in " << mOutputTriggerRecords.size() << "trigger records.";
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggerRecords);
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe}, mOutputHWErrors);
}
//_____________________________________________________________________________
o2::framework::DataProcessorSpec o2::cpv::reco_workflow::getRawToDigitConverterSpec(bool askDISTSTF, bool isPedestal, bool useBadChannelMap, bool useGainCalibration)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("RAWDATA", o2::framework::ConcreteDataTypeMatcher{"CPV", "RAWDATA"}, o2::framework::Lifetime::Optional);
  // receive at least 1 guaranteed input (which will allow to acknowledge the TF)
  if (askDISTSTF) {
    inputs.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }
  if (!isPedestal) {
    inputs.emplace_back("peds", "CPV", "CPV_Pedestals", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CPV/Calib/Pedestals"));
    if (useBadChannelMap) {
      inputs.emplace_back("badmap", "CPV", "CPV_BadMap", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CPV/Calib/BadChannelMap"));
    }
    if (useGainCalibration) {
      inputs.emplace_back("gains", "CPV", "CPV_Gains", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CPV/Calib/Gains"));
    }
  }
  // for BC correction
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CTP/Config/TriggerOffsets"));

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe);
  // note that for cpv we always have stream #0 (i.e. CPV/DIGITS/0)

  return o2::framework::DataProcessorSpec{"CPVRawToDigitConverterSpec",
                                          inputs, // o2::framework::select("A:CPV/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<o2::cpv::reco_workflow::RawToDigitConverterSpec>(isPedestal, useBadChannelMap, useGainCalibration),
                                          o2::framework::Options{}};
}
