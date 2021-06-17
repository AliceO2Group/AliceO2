// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawDataReaderSpec.h

#ifndef O2_FDD_RAWDATAREADERSPEC_H
#define O2_FDD_RAWDATAREADERSPEC_H

#include "DataFormatsFDD/LookUpTable.h"
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
namespace fdd
{
template <typename RawReader>
class RawDataReaderSpec : public Task
{
 public:
  RawDataReaderSpec(const RawReader& rawReader) : mRawReader(rawReader) {}
  RawDataReaderSpec() = default;
  ~RawDataReaderSpec() override = default;
  void init(InitContext& ic) final { o2::fdd::SingleLUT::Instance().printFullMap(); }
  void run(ProcessingContext& pc) final
  {
    // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
    // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
    {
      std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{o2::header::gDataOriginFDD, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
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

    DPLRawParser parser(pc.inputs());
    mRawReader.clear();
    LOG(INFO) << "FDD RawDataReaderSpec";
    uint64_t count = 0;
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      //Proccessing each page
      count++;
      auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mRawReader.process(payload, rdhPtr->linkID, int(0));
    }
    LOG(INFO) << "Pages: " << count;
    mRawReader.accumulateDigits();
    mRawReader.makeSnapshot(pc);
  }
  RawReader mRawReader;
};

template <typename RawReader>
framework::DataProcessorSpec getFDDRawDataReaderSpec(const RawReader& rawReader, bool askSTFDist)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderFDD";
  std::vector<OutputSpec> outputSpec;
  RawReader::prepareOutputSpec(outputSpec);
  std::vector<InputSpec> inputSpec{{"STF", ConcreteDataTypeMatcher{o2::header::gDataOriginFDD, "RAWDATA"}, Lifetime::Optional}};
  if (askSTFDist) {
    inputSpec.emplace_back("STFDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "fdd-datareader-dpl",
    inputSpec,
    outputSpec,
    adaptFromTask<RawDataReaderSpec<RawReader>>(rawReader),
    Options{}};
}

} // namespace fdd
} // namespace o2

#endif /* O2_FDDDATAREADERDPL_H */
