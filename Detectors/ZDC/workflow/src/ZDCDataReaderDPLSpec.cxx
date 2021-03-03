// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

ZDCDataReaderDPLSpec::ZDCDataReaderDPLSpec(const RawReaderZDC& rawReader, const std::string& ccdbURL)
  : mRawReader(rawReader), mccdbHost(ccdbURL)
{
}
  
void ZDCDataReaderDPLSpec::init(InitContext& ic)
{
  o2::ccdb::BasicCCDBManager::instance().setURL(mccdbHost);
}

void ZDCDataReaderDPLSpec::run(ProcessingContext& pc)
{
  DPLRawParser parser(pc.inputs());
  mRawReader.clear();

  //>> update Time-dependent CCDB stuff, at the moment set the moduleconfig only once
  if (!mRawReader.getModuleConfig()) {
    long timeStamp = 0;
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setTimestamp(timeStamp);
    auto moduleConfig = mgr.get<o2::zdc::ModuleConfig>(o2::zdc::CCDBPathConfigModule);
    if (!moduleConfig) {
      LOG(FATAL) << "Cannot module configuratio for timestamp " << timeStamp;
      return;
      LOG(INFO) << "Loaded module configuration for timestamp " << timeStamp;
    }
    mRawReader.setModuleConfig(moduleConfig);
    mRawReader.setTriggerMask();
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


framework::DataProcessorSpec getZDCDataReaderDPLSpec(const RawReaderZDC& rawReader, const std::string& ccdbURL)
{
  LOG(INFO) << "DataProcessorSpec initDataProcSpec() for RawReaderZDC";
  std::vector<OutputSpec> outputSpec;
  RawReaderZDC::prepareOutputSpec(outputSpec);
  return DataProcessorSpec{
    "zdc-datareader-dpl",
    o2::framework::select("TF:ZDC/RAWDATA"),
    outputSpec,
    adaptFromTask<ZDCDataReaderDPLSpec>(rawReader, ccdbURL),
    Options{}};
}
  
} // namespace zdc
} // namespace o2
