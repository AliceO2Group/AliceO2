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

/// @file   FITDataReaderDPLSpec.h

#ifndef O2_FITDATAREADERDPLSPEC_H
#define O2_FITDATAREADERDPLSPEC_H
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/CallbackService.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/SerializationMethods.h"
#include "DPLUtils/DPLRawParser.h"
#include "Framework/InputRecordWalker.h"
#include "DetectorsRaw/RDHUtils.h"
#include <string>
#include <iostream>
#include <vector>
#include <gsl/span>
#include <chrono>
#include "CommonUtils/VerbosityConfig.h"

using namespace o2::framework;

namespace o2
{
namespace fit
{
template <typename RawReaderType>
class FITDataReaderDPLSpec : public Task
{
 public:
  FITDataReaderDPLSpec(const RawReaderType& rawReader, const ConcreteDataMatcher& matcherChMapCCDB, bool isSampledRawData, bool updateCCDB) : mRawReader(rawReader), mMatcherChMapCCDB(matcherChMapCCDB), mIsSampledRawData(isSampledRawData), mUpdateCCDB(updateCCDB) {}
  FITDataReaderDPLSpec() = delete;
  ~FITDataReaderDPLSpec() override = default;
  typedef RawReaderType RawReader_t;
  RawReader_t mRawReader;
  ConcreteDataMatcher mMatcherChMapCCDB; // matcher for Channel map(LUT) from CCDB
  bool mIsSampledRawData;
  bool mUpdateCCDB{true};
  bool mDumpMetrics{false};
  void init(InitContext& ic) final
  {
    auto ccdbUrl = ic.options().get<std::string>("ccdb-path");
    auto lutPath = ic.options().get<std::string>("lut-path");
    mDumpMetrics = ic.options().get<bool>("dump-raw-data-metric");
    if (!ic.options().get<bool>("disable-empty-tf-protection")) {
      mRawReader.enableEmptyTFprotection();
    }
    if (ccdbUrl != "") {
      RawReader_t::LookupTable_t::setCCDBurl(ccdbUrl);
    }
    if (lutPath != "") {
      RawReader_t::LookupTable_t::setLUTpath(lutPath);
    }
    if (!mUpdateCCDB) {
      RawReader_t::LookupTable_t::Instance().printFullMap();
    }
    const auto nReserveVecDig = ic.options().get<int>("reserve-vec-dig");
    const auto nReserveVecChData = ic.options().get<int>("reserve-vec-chdata");
    const auto nReserveVecBuffer = ic.options().get<int>("reserve-vec-buffer");
    const auto nReserveMapDig = ic.options().get<int>("reserve-map-dig");
    if (nReserveVecDig || nReserveVecChData) {
      mRawReader.reserveVecDPL(nReserveVecDig, nReserveVecChData);
    }
    if (nReserveVecBuffer || nReserveMapDig) {
      mRawReader.reserve(nReserveVecBuffer, nReserveMapDig);
    }
  }
  void run(ProcessingContext& pc) final
  {
    // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
    // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
    if (!mIsSampledRawData) {         // do not check 0xDEADBEEF if raw data is sampled
      static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
      std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{mRawReader.mDataOrigin, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
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
    std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{mRawReader.mDataOrigin, mIsSampledRawData ? "SUB_RAWDATA" : o2::header::gDataDescriptionRawData}, Lifetime::Timeframe}};
    DPLRawParser parser(pc.inputs(), filter);
    std::size_t cntDF0{0};        // number of pages with DataFormat=0, padded
    std::size_t cntDF2{0};        // number of pages with DataFormat=2, no padding
    std::size_t cntDF_unknown{0}; // number of pages with unknown DataFormat
    auto start = std::chrono::high_resolution_clock::now();
    if (mUpdateCCDB) {
      pc.inputs().get<typename RawReader_t::LookupTable_t::Table_t*>("channel_map");
      mUpdateCCDB = false;
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      LOG(debug) << "Channel map upload delay: " << duration.count();
    }
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      const o2::header::RDHAny* rdhPtr = nullptr;
      // Proccessing each page
      try {
        rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
        gsl::span<const uint8_t> payload(it.data(), it.size());
        const auto rdhDataFormat = o2::raw::RDHUtils::getDataFormat(rdhPtr);
        if (rdhDataFormat == 0) { // padded
          cntDF0++;
          mRawReader.process(true, payload, o2::raw::RDHUtils::getFEEID(rdhPtr), o2::raw::RDHUtils::getLinkID(rdhPtr), o2::raw::RDHUtils::getEndPointID(rdhPtr));
        } else if (rdhDataFormat == 2) { // no padding
          cntDF2++;
          mRawReader.process(false, payload, o2::raw::RDHUtils::getFEEID(rdhPtr), o2::raw::RDHUtils::getLinkID(rdhPtr), o2::raw::RDHUtils::getEndPointID(rdhPtr));
        } else {
          cntDF_unknown++;
          continue; // or break?
        }
      } catch (std::exception& e) {
        LOG(error) << "Failed to extract RDH, abandoning TF sending dummy output, exception was: " << e.what();
        mRawReader.makeSnapshot(pc); // send empty output
        return;
      }
    }
    mRawReader.accumulateDigits();
    mRawReader.emptyTFprotection();
    mRawReader.makeSnapshot(pc);
    if (mDumpMetrics) {
      mRawReader.dumpRawDataMetrics();
    }
    mRawReader.clear();
    if ((cntDF0 > 0 && cntDF2 > 0) || cntDF_unknown > 0) {
      LOG(error) << "Strange RDH::dataFormat in TF. Number of pages: DF=0 - " << cntDF0 << " , DF=2 - " << cntDF2 << " , DF=unknown - " << cntDF_unknown;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    LOG(debug) << "TF delay: " << duration.count();
  }
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    LOG(info) << "finaliseCCDB";
    if (matcher == mMatcherChMapCCDB) {
      LOG(debug) << "Channel map is updated";
      RawReader_t::LookupTable_t::Instance((const typename RawReader_t::LookupTable_t::Table_t*)obj);
      return;
    }
  }
};

template <typename RawReaderType>
framework::DataProcessorSpec getFITDataReaderDPLSpec(const RawReaderType& rawReader, bool askSTFDist, bool isSubSampled, bool disableDplCcdbFetcher)
{
  std::vector<OutputSpec> outputSpec;
  std::vector<InputSpec> inputSpec{};
  ConcreteDataMatcher matcherChMapCCDB{rawReader.mDataOrigin, RawReaderType::LookupTable_t::sObjectName, 0};
  rawReader.configureOutputSpec(outputSpec);
  if (isSubSampled) {
    inputSpec.push_back({"STF", ConcreteDataTypeMatcher{rawReader.mDataOrigin, "SUB_RAWDATA"}, Lifetime::Sporadic}); // in case if one need to use DataSampler
    askSTFDist = false;
  } else {
    inputSpec.push_back({"STF", ConcreteDataTypeMatcher{rawReader.mDataOrigin, "RAWDATA"}, Lifetime::Optional});
  }
  if (askSTFDist) {
    inputSpec.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  const bool updateCCDB = !disableDplCcdbFetcher;
  if (updateCCDB) {
    inputSpec.emplace_back("channel_map", matcherChMapCCDB, Lifetime::Condition, ccdbParamSpec(RawReaderType::LookupTable_t::sDefaultLUTpath));
  }
  std::string dataProcName = rawReader.mDataOrigin.template as<std::string>();
  std::for_each(dataProcName.begin(), dataProcName.end(), [](char& c) { c = ::tolower(c); });
  dataProcName += "-datareader-dpl";
  LOG(info) << dataProcName;
  return DataProcessorSpec{
    dataProcName,
    inputSpec,
    outputSpec,
    adaptFromTask<FITDataReaderDPLSpec<RawReaderType>>(rawReader, matcherChMapCCDB, isSubSampled, updateCCDB),
    {o2::framework::ConfigParamSpec{"ccdb-path", VariantType::String, "", {"CCDB url which contains LookupTable"}},
     o2::framework::ConfigParamSpec{"lut-path", VariantType::String, "", {"LookupTable path, e.g. FT0/LookupTable"}},
     o2::framework::ConfigParamSpec{"reserve-vec-dig", VariantType::Int, 0, {"Reserve memory for Digit vector, to DPL channel"}},
     o2::framework::ConfigParamSpec{"reserve-vec-chdata", VariantType::Int, 0, {"Reserve memory for ChannelData vector, to DPL channel"}},
     o2::framework::ConfigParamSpec{"reserve-vec-buffer", VariantType::Int, 0, {"Reserve memory for DataBlock vector, buffer for each page"}},
     o2::framework::ConfigParamSpec{"reserve-map-dig", VariantType::Int, 0, {"Reserve memory for Digit map, mapping in RawReader"}},
     o2::framework::ConfigParamSpec{"dump-raw-data-metric", VariantType::Bool, false, {"Dump raw data metrics, for debugging"}},
     o2::framework::ConfigParamSpec{"disable-empty-tf-protection", VariantType::Bool, false, {"Disable empty TF protection. In case of empty payload within TF, only dummy ChannelData object will be sent."}}}};
}

} // namespace fit
} // namespace o2

#endif /* O2_FITDATAREADERDPL_H */
