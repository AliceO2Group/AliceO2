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
namespace fdd
{
template <typename RawReader>
class RawDataReaderSpec : public Task
{
 public:
  RawDataReaderSpec(const RawReader& rawReader) : mRawReader(rawReader) {}
  RawDataReaderSpec() = default;
  ~RawDataReaderSpec() override = default;
  void init(InitContext& ic) final {}
  void run(ProcessingContext& pc) final
  {
    DPLRawParser parser(pc.inputs());
    mRawReader.clear();
    LOG(INFO) << "FDD RawDataReaderSpec";
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
framework::DataProcessorSpec getFDDRawDataReaderSpec(const RawReader& rawReader)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderFDD";
  std::vector<OutputSpec> outputSpec;
  RawReader::prepareOutputSpec(outputSpec);
  return DataProcessorSpec{
    "fdd-datareader-dpl",
    o2::framework::select("TF:FDD/RAWDATA"),
    outputSpec,
    adaptFromTask<RawDataReaderSpec<RawReader>>(rawReader),
    Options{}};
}

} // namespace fdd
} // namespace o2

#endif /* O2_FDDDATAREADERDPL_H */
