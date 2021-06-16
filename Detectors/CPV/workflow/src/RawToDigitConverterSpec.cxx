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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsCPV/CPVBlockHeader.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "CPVReconstruction/RawDecoder.h"
#include "CPVWorkflow/RawToDigitConverterSpec.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "CPVBase/Geometry.h"

using namespace o2::cpv::reco_workflow;

void RawToDigitConverterSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "Initializing RawToDigitConverterSpec...";

  //Read command-line options
  //Pedestal flag true/false
  mIsPedestalData = false;
  if (ctx.options().isSet("pedestal")) {
    mIsPedestalData = ctx.options().get<bool>("pedestal");
  }
  LOG(INFO) << "Pedestal data: " << mIsPedestalData;
  if (mIsPedestalData) { //no calibration for pedestal runs needed
    return;              //skip CCDB initialization for pedestal runs
  }

  //CCDB Url
  std::string ccdbUrl = "localtest";
  if (ctx.options().isSet("ccdb-url")) {
    ccdbUrl = ctx.options().get<std::string>("ccdb-url");
  }
  LOG(INFO) << "CCDB Url: " << ccdbUrl;

  //dummy calibration objects
  if (ccdbUrl.compare("localtest") == 0) { // test default calibration
    mIsUsingCcdbMgr = false;
    mCalibParams = std::make_unique<o2::cpv::CalibParams>(1);
    mBadMap = std::make_unique<o2::cpv::BadChannelMap>(1);
    mPedestals = std::make_unique<o2::cpv::Pedestals>(1);
    LOG(INFO) << "No reading calibration from ccdb requested, using dummy calibration for testing";
    return; //localtest = no reading ccdb
  }

  //init CCDB
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL(ccdbUrl);
  mIsUsingCcdbMgr = ccdbMgr.isHostReachable(); //if host is not reachable we can use only dummy calibration
  if (!mIsUsingCcdbMgr) {
    LOG(ERROR) << "Host " << ccdbUrl << " is not reachable!!!";
    LOG(ERROR) << "Using dummy calibration";
    mCalibParams = std::make_unique<o2::cpv::CalibParams>(1);
    mBadMap = std::make_unique<o2::cpv::BadChannelMap>(1);
    mPedestals = std::make_unique<o2::cpv::Pedestals>(1);
  } else {
    ccdbMgr.setCaching(true);                     //make local cache of remote objects
    ccdbMgr.setLocalObjectValidityChecking(true); //query objects from remote site only when local one is not valid
    LOG(INFO) << "Successfully initializated BasicCCDBManager with caching option";

    //read calibration from ccdb (for now do it only at the beginning of dataprocessing)
    //probably later we can check bad channel map more oftenly
    mCurrentTimeStamp = o2::ccdb::getCurrentTimestamp();
    ccdbMgr.setTimestamp(mCurrentTimeStamp);

    mCalibParams.reset(ccdbMgr.get<o2::cpv::CalibParams>("CPV/Calib/Gains"));
    if (!mCalibParams) {
      LOG(ERROR) << "Cannot get o2::cpv::CalibParams from CCDB. using dummy calibration!";
      mCalibParams = std::make_unique<o2::cpv::CalibParams>(1);
    }
    mBadMap.reset(ccdbMgr.get<o2::cpv::BadChannelMap>("CPV/Calib/BadChannelMap"));
    if (!mBadMap) {
      LOG(ERROR) << "Cannot get o2::cpv::BadChannelMap from CCDB. using dummy calibration!";
      mBadMap = std::make_unique<o2::cpv::BadChannelMap>(1);
    }
    mPedestals.reset(ccdbMgr.get<o2::cpv::Pedestals>("CPV/Calib/Pedestals"));
    if (!mPedestals) {
      LOG(ERROR) << "Cannot get o2::cpv::Pedestals from CCDB. using dummy calibration!";
      mPedestals = std::make_unique<o2::cpv::Pedestals>(1);
    }
    LOG(INFO) << "Task configuration is done.";
  }
}

void RawToDigitConverterSpec::run(framework::ProcessingContext& ctx)
{
  // Cache digits from bunch crossings as the component reads timeframes from many links consecutively
  std::map<o2::InteractionRecord, std::shared_ptr<std::vector<o2::cpv::Digit>>> digitBuffer; // Internal digit buffer
  int firstEntry = 0;
  mOutputHWErrors.clear();

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  std::vector<o2::framework::InputSpec> dummy{o2::framework::InputSpec{"dummy", o2::framework::ConcreteDataMatcher{"CPV", o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
  for (const auto& ref : framework::InputRecordWalker(ctx.inputs(), dummy)) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (dh->payloadSize == 0) { // send empty output
      LOG(INFO) << "Sending empty output due to data type input with 0xDEADBEEF";
      mOutputDigits.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
      mOutputTriggerRecords.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggerRecords);
      mOutputHWErrors.clear();
      ctx.outputs().snapshot(o2::framework::Output{"CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe}, mOutputHWErrors);
      return; //empty TF, nothing to process
    }
  }

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
        LOG(ERROR) << "Raw decoding error " << (int)e;
        //add error list
        //RawErrorType_t is defined in O2/Detectors/CPV/reconstruction/include/CPVReconstruction/RawReaderMemory.h
        //RawDecoderError(short c, short d, short g, short p, RawErrorType_t e)
        mOutputHWErrors.emplace_back(25, 0, 0, 0, e); //Put general errors to non-existing ccId 25
        //if problem in header, abandon this page
        if (e == RawErrorType_t::kRDH_DECODING) {
          break;
        }
        //if problem in payload, try to continue
        continue;
      }
      auto& rdh = rawreader.getRawHeader();
      auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(rdh);
      auto mod = o2::raw::RDHUtils::getLinkID(rdh) + 2; //link=0,1,2 -> mod=2,3,4
      //for now all modules are written to one LinkID
      if (mod > o2::cpv::Geometry::kNMod || mod < 2) { //only 3 correct modules:2,3,4
        LOG(ERROR) << "module=" << mod << "do not exist";
        mOutputHWErrors.emplace_back(25, mod, 0, 0, kRDH_INVALID); //Add non-existing modules to non-existing ccId 25 and dilogic = mod
        continue;                                                  //skip STU mod
      }
      o2::cpv::RawDecoder decoder(rawreader);
      RawErrorType_t err = decoder.decode();

      if (!(err == kOK || err == kOK_NO_PAYLOAD)) {
        //TODO handle severe errors
        //TODO: probably careful conversion of decoder errors to Fitter errors?
        mOutputHWErrors.emplace_back(25, mod, 0, 0, err); //assign general RDH errors to non-existing ccId 25 and dilogic = mod
      }

      std::shared_ptr<std::vector<o2::cpv::Digit>> currentDigitContainer;
      auto digilets = decoder.getDigits();
      if (digilets.empty()) { //no digits -> continue to next pages
        continue;
      }
      o2::InteractionRecord currentIR(0, triggerOrbit); //(bc, orbit)
      // Loop over all the BCs
      for (auto itBCRecords : decoder.getBCRecords()) {
        currentIR.bc = itBCRecords.bc;
        for (int iDig = itBCRecords.firstDigit; iDig <= itBCRecords.lastDigit; iDig++) {
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
          //if we deal with non-pedestal data?
          if (!mIsPedestalData) { //not a pedestal data
            //test bad map
            if (mBadMap->isChannelGood(absId)) {
              //we need to subtract pedestal from amplidute and calibrate it
              float amp = mCalibParams->getGain(absId) * (ac.Charge - mPedestals->getPedestal(absId));
              if (amp > 0) {
                currentDigitContainer->emplace_back(absId, amp, -1);
              }
            }
          } else { //pedestal data, no calibration needed.
            currentDigitContainer->emplace_back(absId, (float)ac.Charge, -1);
          }
        }
      }
      //Check and send list of hwErrors
      for (auto& er : decoder.getErrors()) {
        mOutputHWErrors.push_back(er);
      }
    } //RawReader::hasNext
  }

  // Loop over BCs, sort digits with increasing digit ID and write to output containers
  mOutputDigits.clear();
  mOutputTriggerRecords.clear();
  for (auto [bc, digits] : digitBuffer) {
    int prevDigitSize = mOutputDigits.size();
    if (digits->size()) {
      // Sort digits according to digit ID
      std::sort(digits->begin(), digits->end(), [](o2::cpv::Digit& lhs, o2::cpv::Digit& rhs) { return lhs.getAbsId() < rhs.getAbsId(); });

      for (auto digit : *digits) {
        mOutputDigits.push_back(digit);
      }
    }

    mOutputTriggerRecords.emplace_back(bc, prevDigitSize, mOutputDigits.size() - prevDigitSize);
  }
  digitBuffer.clear();

  LOG(INFO) << "[CPVRawToDigitConverter - run] Writing " << mOutputDigits.size() << " digits ...";
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggerRecords);
  ctx.outputs().snapshot(o2::framework::Output{"CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe}, mOutputHWErrors);
}

o2::framework::DataProcessorSpec o2::cpv::reco_workflow::getRawToDigitConverterSpec(bool askDISTSTF)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("RAWDATA", o2::framework::ConcreteDataTypeMatcher{"CPV", "RAWDATA"}, o2::framework::Lifetime::Optional);
  //receive at least 1 guaranteed input (which will allow to acknowledge the TF)
  if (askDISTSTF) {
    inputs.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe);
  //note that for cpv we always have stream #0 (i.e. CPV/DIGITS/0)

  return o2::framework::DataProcessorSpec{"CPVRawToDigitConverterSpec",
                                          inputs, // o2::framework::select("A:CPV/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<o2::cpv::reco_workflow::RawToDigitConverterSpec>(),
                                          o2::framework::Options{
                                            {"pedestal", o2::framework::VariantType::Bool, false, {"If true then do not subtract pedestals from digits"}},
                                            {"ccdb-url", o2::framework::VariantType::String, "http://ccdb-test.cern.ch:8080", {"CCDB Url"}},
                                          }};
}
