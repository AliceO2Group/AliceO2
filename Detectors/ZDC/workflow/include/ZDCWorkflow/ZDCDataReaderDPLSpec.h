// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ZDCDataReaderDPLSpec.h

#ifndef O2_ZDCDATAREADERDPLSPEC_H
#define O2_ZDCDATAREADERDPLSPEC_H

#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
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
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#
#include <iostream>
#include <vector>
#include <gsl/span>
using namespace o2::framework;

namespace o2
{
namespace zdc
{
template <typename RawReader>
class ZDCDataReaderDPLSpec : public Task
{
 public:
  ZDCDataReaderDPLSpec(const RawReader& rawReader) : mRawReader(rawReader) {}
  ZDCDataReaderDPLSpec() = default;
  ~ZDCDataReaderDPLSpec() override = default;
  void init(InitContext& ic) final {}
  void run(ProcessingContext& pc) final
  {
    DPLRawParser parser(pc.inputs());
    mRawReader.clear();
    long timeStamp = 0;
    std::string ccdbHost = "http://ccdb-test.cern.ch:8080";
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL(ccdbHost);
    if (timeStamp == mgr.getTimestamp()) {
      return;
    }
    mgr.setTimestamp(timeStamp);
    auto moduleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
    if (!moduleConfig) {
      LOG(FATAL) << "Cannot module configuratio for timestamp " << timeStamp;
      return;
    }
    LOG(INFO) << "Loaded module configuration for timestamp " << timeStamp;
    mRawReader.setModuleConfig(moduleConfig);
    LOG(INFO) << "ZDCDataReaderDPLSpec";
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
framework::DataProcessorSpec getZDCDataReaderDPLSpec(const RawReader& rawReader)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderZDC";
  std::vector<OutputSpec> outputSpec;
  RawReader::prepareOutputSpec(outputSpec);
  return DataProcessorSpec{
    "zdc-datareader-dpl",
    o2::framework::select("TF:ZDC/RAWDATA"),
    outputSpec,
    adaptFromTask<ZDCDataReaderDPLSpec<RawReader>>(rawReader),
    Options{}};
}

} // namespace zdc
} // namespace o2

#endif /* O2_ZDCDATAREADERDPL_H */
