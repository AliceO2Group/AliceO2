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

/// @file   FV0DataReaderDPLSpec.h

#ifndef O2_FV0DATAREADERDPLSPEC_H
#define O2_FV0DATAREADERDPLSPEC_H
#include "DataFormatsFV0/LookUpTable.h"
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
#include <iostream>
#include <vector>
#include <gsl/span>
using namespace o2::framework;

namespace o2
{
namespace fv0
{
template <typename RawReader>
class FV0DataReaderDPLSpec : public Task
{
 public:
  FV0DataReaderDPLSpec(const RawReader& rawReader) : mRawReader(rawReader) {}
  FV0DataReaderDPLSpec() = default;
  ~FV0DataReaderDPLSpec() override = default;
  typedef RawReader RawReader_t;
  void init(InitContext& ic) final { o2::fv0::SingleLUT::Instance().printFullMap(); }
  void run(ProcessingContext& pc) final
  {
    // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
    // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
    {
      std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{o2::header::gDataOriginFV0, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
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
    std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{o2::header::gDataOriginFV0, o2::header::gDataDescriptionRawData}, Lifetime::Timeframe}};
    DPLRawParser parser(pc.inputs(), filter);
    std::size_t count = 0;
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      //Proccessing each page
      count++;
      auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mRawReader.process(payload, rdhPtr->linkID, rdhPtr->endPointID);
    }
    LOG(INFO) << "Pages: " << count;
    mRawReader.accumulateDigits();
    mRawReader.makeSnapshot(pc);
    mRawReader.clear();
  }
  RawReader_t mRawReader;
};

template <typename RawReader>
framework::DataProcessorSpec getFV0DataReaderDPLSpec(const RawReader& rawReader, bool askSTFDist)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderFV0";
  std::vector<OutputSpec> outputSpec;
  RawReader::prepareOutputSpec(outputSpec);
  std::vector<InputSpec> inputSpec{{"STF", ConcreteDataTypeMatcher{o2::header::gDataOriginFV0, "RAWDATA"}, Lifetime::Optional}};
  if (askSTFDist) {
    inputSpec.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "fv0-datareader-dpl",
    inputSpec,
    outputSpec,
    adaptFromTask<FV0DataReaderDPLSpec<RawReader>>(rawReader),
    Options{}};
}

} // namespace fv0
} // namespace o2

#endif /* O2_FV0DATAREADERDPL_H */
