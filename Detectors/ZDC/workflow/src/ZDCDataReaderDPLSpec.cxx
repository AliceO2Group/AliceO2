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
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{o2::header::gDataOriginZDC, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(pc.inputs(), dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (dh->payloadSize == 0) {
        LOGP(WARNING, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
        mRawReader.makeSnapshot(pc); // send empty output
        return;
      }
    }
  }

  DPLRawParser parser(pc.inputs(), o2::framework::select("zdc:ZDC/RAWDATA"));

  //>> update Time-dependent CCDB stuff, at the moment set the moduleconfig only once
  if (!mRawReader.getModuleConfig()) {
    long timeStamp = 0;
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setTimestamp(timeStamp);
    auto moduleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
    if (!moduleConfig) {
      LOG(FATAL) << "Cannot module configuratio for timestamp " << timeStamp;
      return;
    } else {
      LOG(INFO) << "Loaded module configuration for timestamp " << timeStamp;
    }
    mRawReader.setModuleConfig(moduleConfig);
    mRawReader.setTriggerMask();
    mRawReader.setVerifyTrigger(mVerifyTrigger);
    LOG(INFO) << "Check of trigger condition during conversion is " << (mVerifyTrigger ? "ON" : "OFF");
  }
  uint64_t count = 0;
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    //Proccessing each page
    count++;
    auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
    gsl::span<const uint8_t> payload(it.data(), it.size());
    mRawReader.processBinaryData(payload, rdhPtr->linkID);
  }
  LOG(INFO) << "Pages: " << count;
  mRawReader.accumulateDigits();
  mRawReader.makeSnapshot(pc);
}

framework::DataProcessorSpec getZDCDataReaderDPLSpec(const RawReaderZDC& rawReader, const bool verifyTrigger, const bool askSTFDist)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderZDC";
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
    Options{{"ccdb-url", o2::framework::VariantType::String, "http://ccdb-test.cern.ch:8080", {"CCDB Url"}}}};
}
} // namespace zdc
} // namespace o2
