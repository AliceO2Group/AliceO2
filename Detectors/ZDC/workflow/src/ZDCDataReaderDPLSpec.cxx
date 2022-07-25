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

/// @file   ZDCDataReaderDPLSpec.cxx

#include "ZDCWorkflow/ZDCDataReaderDPLSpec.h"
#include "CommonUtils/VerbosityConfig.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

ZDCDataReaderDPLSpec::ZDCDataReaderDPLSpec(const RawReaderZDC& rawReader, const bool verifyTrigger)
  : mRawReader(rawReader), mVerifyTrigger(verifyTrigger)
{
}

void ZDCDataReaderDPLSpec::init(InitContext& ic)
{
  mccdbHost = ic.options().get<std::string>("ccdb-url");
  o2::ccdb::BasicCCDBManager::instance().setURL(mccdbHost);
}

void ZDCDataReaderDPLSpec::run(ProcessingContext& pc)
{
  mRawReader.clear();

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{o2::header::gDataOriginZDC, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(pc.inputs(), dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        mRawReader.makeSnapshot(pc); // send empty output
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }

  DPLRawParser parser(pc.inputs(), o2::framework::select("zdc:ZDC/RAWDATA"));

  //>> update Time-dependent CCDB stuff, at the moment set the moduleconfig only once
  if (!mRawReader.getModuleConfig()) {
    /*long timeStamp = 0; // TIMESTAMP SHOULD NOT BE 0
    mgr.setTimestamp(timeStamp);*/
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    auto moduleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
    if (!moduleConfig) {
      LOG(fatal) << "Cannot retrieve module configuration from " << o2::zdc::CCDBPathConfigModule << " for timestamp " << mgr.getTimestamp();
      return;
    } else {
      LOG(info) << "Loaded module configuration for timestamp " << mgr.getTimestamp();
    }
    mRawReader.setModuleConfig(moduleConfig);
    mRawReader.setTriggerMask();
    mRawReader.setVerifyTrigger(mVerifyTrigger);
    LOG(info) << "Check of trigger condition during conversion is " << (mVerifyTrigger ? "ON" : "OFF");
  }

  uint64_t count = 0;
  static uint64_t nErr[3] = {0};
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    // Processing each page
    auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
    if (rdhPtr == nullptr) {
      nErr[0]++;
      if (nErr[0] < 5) {
        LOG(warning) << "ZDCDataReaderDPLSpec::run - Missing RAWDataHeader on page " << count;
      } else if (nErr[0] == 5) {
        LOG(warning) << "ZDCDataReaderDPLSpec::run - Missing RAWDataHeader on page " << count << " suppressing further messages";
      }
    } else {
      if (it.data() == nullptr) {
        nErr[1]++;
      } else if (it.size() == 0) {
        nErr[2]++;
      } else {
        gsl::span<const uint8_t> payload(it.data(), it.size());
        mRawReader.processBinaryData(payload, rdhPtr->linkID);
      }
    }
    count++;
  }
  LOG(info) << "ZDCDataReaderDPLSpec::run processed pages: " << count;
  if (nErr[0] > 0) {
    LOG(warning) << "ZDCDataReaderDPLSpec::run - Missing RAWDataHeader occurrences " << nErr[0];
  }
  if (nErr[1] > 0) {
    LOG(warning) << "ZDCDataReaderDPLSpec::run - Null payload pointer occurrences " << nErr[1];
  }
  if (nErr[2] > 0) {
    LOG(warning) << "ZDCDataReaderDPLSpec::run - No payload occurrences " << nErr[2];
  }
  if (nErr[0] == 0) {
    mRawReader.accumulateDigits();
  } else {
    LOG(warning) << "Not sending output ";
  }
  mRawReader.makeSnapshot(pc);
}

framework::DataProcessorSpec getZDCDataReaderDPLSpec(const RawReaderZDC& rawReader, const bool verifyTrigger, const bool askSTFDist)
{
  LOG(info) << "DataProcessorSpec initDataProcSpec() for RawReaderZDC";
  std::vector<OutputSpec> outputSpec;
  RawReaderZDC::prepareOutputSpec(outputSpec);
  std::vector<InputSpec> inputSpec{{"STF", ConcreteDataTypeMatcher{o2::header::gDataOriginZDC, "RAWDATA"}, Lifetime::Optional}};
  if (askSTFDist) {
    inputSpec.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "zdc-datareader-dpl",
    inputSpec,
    outputSpec,
    adaptFromTask<ZDCDataReaderDPLSpec>(rawReader, verifyTrigger),
    Options{{"ccdb-url", o2::framework::VariantType::String, o2::base::NameConf::getCCDBServer(), {"CCDB Url"}}}};
}
} // namespace zdc
} // namespace o2
