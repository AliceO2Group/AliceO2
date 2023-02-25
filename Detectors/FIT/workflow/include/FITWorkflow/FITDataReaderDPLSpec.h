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
  FITDataReaderDPLSpec(const RawReaderType& rawReader) : mRawReader(rawReader) {}
  FITDataReaderDPLSpec() = delete;
  ~FITDataReaderDPLSpec() override = default;
  typedef RawReaderType RawReader_t;
  RawReader_t mRawReader;
  void init(InitContext& ic) final
  {
    auto ccdbUrl = ic.options().get<std::string>("ccdb-path");
    auto lutPath = ic.options().get<std::string>("lut-path");
    if (!ic.options().get<bool>("disable-empty-tf-protection")) {
      mRawReader.enableEmptyTFprotection();
    }
    if (ccdbUrl != "") {
      RawReader_t::LookupTable_t::setCCDBurl(ccdbUrl);
    }
    if (lutPath != "") {
      RawReader_t::LookupTable_t::setLUTpath(lutPath);
    }
    RawReader_t::LookupTable_t::Instance().printFullMap();
    auto nReserveVecDig = ic.options().get<int>("reserve-vec-dig");
    auto nReserveVecChData = ic.options().get<int>("reserve-vec-chdata");
    auto nReserveVecBuffer = ic.options().get<int>("reserve-vec-buffer");
    auto nReserveMapDig = ic.options().get<int>("reserve-map-dig");
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
    {
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
    std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{mRawReader.mDataOrigin, o2::header::gDataDescriptionRawData}, Lifetime::Timeframe}};
    DPLRawParser parser(pc.inputs(), filter);
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      const o2::header::RDHAny* rdh = nullptr;
      // Proccessing each page
      try {
        rdh = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
        gsl::span<const uint8_t> payload(it.data(), it.size());
        mRawReader.process(payload, o2::raw::RDHUtils::getLinkID(rdh), o2::raw::RDHUtils::getEndPointID(rdh));
      } catch (std::exception& e) {
        LOG(error) << "Failed to extract RDH, abandoning TF sending dummy output, exception was: " << e.what();
        mRawReader.makeSnapshot(pc); // send empty output
        return;
      }
    }
    mRawReader.accumulateDigits();
    mRawReader.emptyTFprotection();
    mRawReader.makeSnapshot(pc);
    mRawReader.clear();
  }
};

template <typename RawReaderType>
framework::DataProcessorSpec getFITDataReaderDPLSpec(const RawReaderType& rawReader, bool askSTFDist)
{
  std::vector<OutputSpec> outputSpec;
  rawReader.configureOutputSpec(outputSpec);
  std::vector<InputSpec> inputSpec{{"STF", ConcreteDataTypeMatcher{rawReader.mDataOrigin, "RAWDATA"}, Lifetime::Optional}};
  if (askSTFDist) {
    inputSpec.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  std::string dataProcName = rawReader.mDataOrigin.template as<std::string>();
  std::for_each(dataProcName.begin(), dataProcName.end(), [](char& c) { c = ::tolower(c); });
  dataProcName += "-datareader-dpl";
  LOG(info) << dataProcName;
  return DataProcessorSpec{
    dataProcName,
    inputSpec,
    outputSpec,
    adaptFromTask<FITDataReaderDPLSpec<RawReaderType>>(rawReader),
    {o2::framework::ConfigParamSpec{"ccdb-path", VariantType::String, "", {"CCDB url which contains LookupTable"}},
     o2::framework::ConfigParamSpec{"lut-path", VariantType::String, "", {"LookupTable path, e.g. FT0/LookupTable"}},
     o2::framework::ConfigParamSpec{"reserve-vec-dig", VariantType::Int, 0, {"Reserve memory for Digit vector, to DPL channel"}},
     o2::framework::ConfigParamSpec{"reserve-vec-chdata", VariantType::Int, 0, {"Reserve memory for ChannelData vector, to DPL channel"}},
     o2::framework::ConfigParamSpec{"reserve-vec-buffer", VariantType::Int, 0, {"Reserve memory for DataBlock vector, buffer for each page"}},
     o2::framework::ConfigParamSpec{"reserve-map-dig", VariantType::Int, 0, {"Reserve memory for Digit map, mapping in RawReader"}},
     o2::framework::ConfigParamSpec{"disable-empty-tf-protection", VariantType::Bool, false, {"Disable empty TF protection. In case of empty payload within TF, only dummy ChannelData object will be sent."}}}};
}

} // namespace fit
} // namespace o2

#endif /* O2_FITDATAREADERDPL_H */
