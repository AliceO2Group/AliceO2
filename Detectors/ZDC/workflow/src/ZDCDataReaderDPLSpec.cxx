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
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

ZDCDataReaderDPLSpec::ZDCDataReaderDPLSpec(const RawReaderZDC& rawReader) : mRawReader(rawReader)
{
}

void ZDCDataReaderDPLSpec::init(InitContext& ic)
{
  mVerbosity = ic.options().get<int>("log-level");
  // 0: minimal output
  // 1: event summary per channel
  // 2: debug inconsistencies
  // 3: dump of associated input data
  // 4: dump of raw input data
}

void ZDCDataReaderDPLSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
}

void ZDCDataReaderDPLSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "MODULECONFIG", 0)) {
    mRawReader.setModuleConfig((const o2::zdc::ModuleConfig*)obj);
    mRawReader.setTriggerMask();
  }
}

void ZDCDataReaderDPLSpec::run(ProcessingContext& pc)
{
  mRawReader.clear();
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mRawReader.setVerbosity(mVerbosity);
  }

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

  uint64_t count = 0;
  static uint64_t nErr[3] = {0};
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    // Processing each page
    auto rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
    if (rdhPtr == nullptr || !o2::raw::RDHUtils::checkRDH(rdhPtr, true)) {
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
        auto lid = o2::raw::RDHUtils::getLinkID(rdhPtr);
#ifdef O2_ZDC_DEBUG
        LOG(info) << count << " processBinaryData: size=" << it.size() << " link=" << lid;
#endif
        mRawReader.processBinaryData(payload, lid);
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

framework::DataProcessorSpec getZDCDataReaderDPLSpec(const RawReaderZDC& rawReader, const bool askSTFDist)
{
  LOG(info) << "DataProcessorSpec initDataProcSpec() for RawReaderZDC";
  std::vector<OutputSpec> outputSpec;
  RawReaderZDC::prepareOutputSpec(outputSpec);
  std::vector<InputSpec> inputSpec{{"STF", ConcreteDataTypeMatcher{o2::header::gDataOriginZDC, "RAWDATA"}, Lifetime::Optional}};
  inputSpec.emplace_back("moduleconfig", "ZDC", "MODULECONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathConfigModule.data()));
  if (askSTFDist) {
    inputSpec.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "zdc-datareader-dpl",
    inputSpec,
    outputSpec,
    adaptFromTask<ZDCDataReaderDPLSpec>(rawReader),
    Options{{"ccdb-url", o2::framework::VariantType::String, o2::base::NameConf::getCCDBServer(), {"CCDB Url"}},
            {"log-level", o2::framework::VariantType::Int, 0, {"ZDC data reader verbosity level"}}}};
}
} // namespace zdc
} // namespace o2
