// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <string>
#include <iostream>
#include <vector>
#include <gsl/span>
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
  void init(InitContext& ic) final { RawReader_t::LookupTable_t::Instance().printFullMap(); }
  void run(ProcessingContext& pc) final
  {
    // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
    // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
    {
      std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{mRawReader.mDataOrigin, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
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
    std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{mRawReader.mDataOrigin, o2::header::gDataDescriptionRawData}, Lifetime::Timeframe}};
    DPLRawParser parser(pc.inputs(), filter);
    std::size_t count = 0;
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      //Proccessing each page
      count++;
      auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mRawReader.process(payload, int(rdhPtr->linkID), int(rdhPtr->endPointID));
    }
    LOG(INFO) << "Pages: " << count;
    mRawReader.accumulateDigits();
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
  LOG(INFO) << dataProcName;
  return DataProcessorSpec{
    dataProcName,
    inputSpec,
    outputSpec,
    adaptFromTask<FITDataReaderDPLSpec<RawReaderType>>(rawReader),
    Options{}};
}

} // namespace fit
} // namespace o2

#endif /* O2_FITDATAREADERDPL_H */
