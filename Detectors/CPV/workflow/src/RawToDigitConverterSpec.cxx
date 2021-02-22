// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "CCDB/CcdbApi.h"
#include "CPVBase/CPVSimParams.h"
#include "CPVBase/Geometry.h"

using namespace o2::cpv::reco_workflow;

void RawToDigitConverterSpec::init(framework::InitContext& ctx)
{
  mDDL = ctx.options().get<int>("DDL");
  LOG(DEBUG) << "Initialize converter ";
}

void RawToDigitConverterSpec::run(framework::ProcessingContext& ctx)
{
  // Cache digits from bunch crossings as the component reads timeframes from many links consecutively
  std::map<o2::InteractionRecord, std::shared_ptr<std::vector<o2::cpv::Digit>>> digitBuffer; // Internal digit buffer
  int firstEntry = 0;
  mOutputHWErrors.clear();

  if (!mCalibParams) {
    if (o2::cpv::CPVSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mCalibParams = std::make_unique<o2::cpv::CalibParams>(1); // test default calibration
      LOG(INFO) << "No reading calibration from ccdb requested, set default";
    } else {
      LOG(INFO) << "Getting calibration object from ccdb";
      //TODO: configuring ccdb address from config file, readign proper calibration/BadMap and updateing if necessary
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata;
      ccdb.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
      // auto tr = triggerbranch.begin();
      // double eventTime = -1;
      // if(tr!=triggerbranch.end()){
      //   eventTime = (*tr).getBCData().getTimeNS() ;
      // }
      // mCalibParams = ccdb.retrieveFromTFileAny<o2::cpv::CalibParams>("CPV/Calib", metadata, eventTime);
      // if (!mCalibParams) {
      //   LOG(FATAL) << "Can not get calibration object from ccdb";
      // }
    }
  }

  if (!mBadMap) {
    if (o2::cpv::CPVSimParams::Instance().mCCDBPath.compare("localtest") == 0) {
      mBadMap = std::make_unique<o2::cpv::BadChannelMap>(1); // test default calibration
      LOG(INFO) << "No reading bad map from ccdb requested, set default";
    } else {
      LOG(INFO) << "Getting bad map object from ccdb";
      o2::ccdb::CcdbApi ccdb;
      std::map<std::string, std::string> metadata;
      ccdb.init("http://ccdb-test.cern.ch:8080"); // or http://localhost:8080 for a local installation
      // auto tr = triggerbranch.begin();
      // double eventTime = -1;
      // if(tr!=triggerbranch.end()){
      //   eventTime = (*tr).getBCData().getTimeNS() ;
      // }
      // mBadMap = ccdb.retrieveFromTFileAny<o2::cpv::BadChannelMap>("CPV/BadMap", metadata, eventTime);
      // if (!mBadMap) {
      //   LOG(FATAL) << "Can not get bad map object from ccdb";
      // }
    }
  }

  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {

    // enum RawErrorType_t {
    //   kOK,                       ///< NoError
    //   kNO_PAYLOAD,               ///< No payload per ddl
    //   kHEADER_DECODING,
    //   kPAGE_NOTFOUND,
    //   kPAYLOAD_DECODING,
    //   kHEADER_INVALID,
    //   kRCU_TRAILER_ERROR,        ///< RCU trailer cannot be decoded or invalid
    //   kRCU_VERSION_ERROR,        ///< RCU trailer version not matching with the version in the raw header
    //   kRCU_TRAILER_SIZE_ERROR,   ///< RCU trailer size length
    //   kSEGMENT_HEADER_ERROR,
    //   kROW_HEADER_ERROR,
    //   kEOE_HEADER_ERROR,
    //   kPADERROR,
    //   kPadAddress
    // };

    o2::cpv::RawReaderMemory rawreader(o2::framework::DataRefUtils::as<const char>(rawData));
    // loop over all the DMA pages
    while (rawreader.hasNext()) {
      try {
        rawreader.next();
      } catch (RawErrorType_t e) {
        LOG(ERROR) << "Raw decoding error " << (int)e;
        //add error list
        mOutputHWErrors.emplace_back(5, 0, 0, 0, e); //Put general errors to non-existing DDL5
        //if problem in header, abandon this page
        if (e == RawErrorType_t::kPAGE_NOTFOUND ||
            e == RawErrorType_t::kHEADER_DECODING ||
            e == RawErrorType_t::kHEADER_INVALID) {
          break;
        }
        //if problem in payload, try to continue
        continue;
      }
      auto& header = rawreader.getRawHeader();
      auto triggerBC = o2::raw::RDHUtils::getTriggerBC(header);
      auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(header);
      // auto ddl = o2::raw::RDHUtils::getFEEID(header);
      auto mod = o2::raw::RDHUtils::getLinkID(header) + 2; //ddl=0,1,2 -> mod=2,3,4
      // if(ddl != mDDL){
      //   LOG(ERROR) << "DDL from header "<< ddl << " != configured DDL=" << mDDL;
      // }

      o2::InteractionRecord currentIR(triggerBC, triggerOrbit);
      std::shared_ptr<std::vector<o2::cpv::Digit>> currentDigitContainer;
      auto found = digitBuffer.find(currentIR);
      if (found == digitBuffer.end()) {
        currentDigitContainer = std::make_shared<std::vector<o2::cpv::Digit>>();
        digitBuffer[currentIR] = currentDigitContainer;
      } else {
        currentDigitContainer = found->second;
      }
      //
      if (mod > o2::cpv::Geometry::kNMod) { //only 3 correct modules:2,3,4
        LOG(ERROR) << "module=" << mod << "do not exist";
        mOutputHWErrors.emplace_back(6, mod, 0, 0, kHEADER_INVALID); //Add non-existing DDL as DDL 5
        continue;                                                    //skip STU mod
      }
      // use the altro decoder to decode the raw data, and extract the RCU trailer
      o2::cpv::RawDecoder decoder(rawreader);
      RawErrorType_t err = decoder.decode();

      if (err != kOK) {
        //TODO handle severe errors
        //TODO: probably careful conversion of decoder errors to Fitter errors?
        mOutputHWErrors.emplace_back(mod, 1, 0, 0, err); //assign general header errors to non-existing FEE 16
      }
      // Loop over all the channels
      for (uint32_t adch : decoder.getDigits()) {
        AddressCharge ac = {adch};
        unsigned short absId = ac.Address;
        //test bad map
        if (mBadMap->isChannelGood(absId)) {
          if (ac.Charge > o2::cpv::CPVSimParams::Instance().mZSthreshold) {
            float amp = mCalibParams->getGain(absId) * ac.Charge;
            currentDigitContainer->emplace_back(absId, amp, -1);
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

o2::framework::DataProcessorSpec o2::cpv::reco_workflow::getRawToDigitConverterSpec()
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("RAWDATA", o2::framework::ConcreteDataTypeMatcher{"CPV", "RAWDATA"}, o2::framework::Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("CPV", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "DIGITTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("CPV", "RAWHWERRORS", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"CPVRawToDigitConverterSpec",
                                          inputs, // o2::framework::select("A:CPV/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<o2::cpv::reco_workflow::RawToDigitConverterSpec>(),
                                          o2::framework::Options{
                                            {"pedestal", o2::framework::VariantType::String, "off", {"Analyze as pedestal run on/off"}},
                                            {"DDL", o2::framework::VariantType::String, "0", {"DDL id to read"}},
                                          }};
}
