// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0DataReaderDPLSpec.h

#ifndef O2_FT0DATAREADERDPLSPEC_H
#define O2_FT0DATAREADERDPLSPEC_H

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

#include <iostream>
#include <vector>
#include <gsl/span>
using namespace o2::framework;

namespace o2
{
namespace ft0
{
template <typename RawReader>
class FT0DataReaderDPLSpec : public Task
{
 public:
  FT0DataReaderDPLSpec(const RawReader& rawReader) : mRawReader(rawReader) {}
  FT0DataReaderDPLSpec() = default;
  ~FT0DataReaderDPLSpec() override = default;
  void init(InitContext& ic) final {}
  void run(ProcessingContext& pc) final
  {
    DPLRawParser parser(pc.inputs());
    mRawReader.clear();
    LOG(INFO) << "FT0DataReaderDPLSpec";
    uint64_t count = 0;
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      //Proccessing each page
      count++;
      auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mRawReader.process(rdhPtr->linkID, payload);
    }
    LOG(INFO) << "Pages: " << count;
    mRawReader.accumulateDigits();
    mRawReader.makeSnapshot(pc);
  }
  RawReader mRawReader;
};

template <typename RawReader>
framework::DataProcessorSpec getFT0DataReaderDPLSpec(const RawReader& rawReader)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderFT0";
  std::vector<OutputSpec> outputSpec;
  RawReader::prepareOutputSpec(outputSpec);
  return DataProcessorSpec{
    "ft0-datareader-dpl",
    o2::framework::select("TF:FT0/RAWDATA"),
    outputSpec,
    adaptFromTask<FT0DataReaderDPLSpec<RawReader>>(rawReader),
    Options{}};
}

} // namespace ft0
} // namespace o2

#endif /* O2_FT0DATAREADERDPL_H */
