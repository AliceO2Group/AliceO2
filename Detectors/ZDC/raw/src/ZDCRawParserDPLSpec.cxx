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

/// @file   ZDCRawParserEPNSpec.cxx
/// @brief  ZDC baseline calibration
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include <gsl/span>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"
#include "DPLUtils/DPLRawParser.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/WorkflowSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "DataFormatsZDC/RawEventData.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCBase/ModuleConfig.h"
#include "ZDCRaw/ZDCRawParserDPLSpec.h"
#include "ZDCSimulation/Digits2Raw.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

ZDCRawParserDPLSpec::ZDCRawParserDPLSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

ZDCRawParserDPLSpec::ZDCRawParserDPLSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void ZDCRawParserDPLSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("log-level");
  mWorker.setVerbosity(mVerbosity);
}

void ZDCRawParserDPLSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mWorker.init();
    mWorker.setVerbosity(mVerbosity);
    mInitialized = true;
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
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
        // retrieving the raw pointer of the page
        auto const* raw = it.raw();
        // retrieving payload pointer of the page
        auto const* payload = it.data();
        // size of payload
        size_t payloadSize = it.size();
        // offset of payload in the raw page
        size_t offset = it.offset();
        int dataFormat = o2::raw::RDHUtils::getDataFormat(rdhPtr);
#ifdef O2_ZDC_DEBUG
        int linkID = o2::raw::RDHUtils::getLinkID(rdhPtr);
        LOG(info) << count << " ZDCRawParserDPLSpec::run: fmt=" << dataFormat << " size=" << it.size() << " link=" << linkID;
#endif
        if (dataFormat == 2) {
          for (int32_t ip = 0; (ip + PayloadPerGBTW) <= payloadSize; ip += PayloadPerGBTW) {
            const uint32_t* gbtw = (const uint32_t*)&payload[ip];
#ifdef O2_ZDC_DEBUG
            o2::zdc::Digits2Raw::print_gbt_word(gbtw);
#endif
            if (gbtw[0] != 0xffffffff || gbtw[1] != 0xffffffff || (gbtw[2] & 0xffff) != 0xffff) {
              mWorker.processWord(gbtw);
            }
          }
        } else if (dataFormat == 0) {
          for (int32_t ip = 0; ip < payloadSize; ip += NBPerGBTW) {
#ifdef O2_ZDC_DEBUG
            o2::zdc::Digits2Raw::print_gbt_word((const uint32_t*)&payload[ip]);
#endif
            mWorker.processWord((const uint32_t*)&payload[ip]);
          }
        } else {
          LOG(error) << "ZDCDataReaderDPLSpec::run - Unsupported DataFormat " << dataFormat;
        }
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
}

void ZDCRawParserDPLSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.write();
  mTimer.Stop();
  LOGF(info, "ZDC RAW data parsing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getZDCRawParserDPLSpec()
{
  using device = o2::zdc::ZDCRawParserDPLSpec;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("raw", o2::framework::ConcreteDataTypeMatcher{"ZDC", "RAWDATA"}, Lifetime::Optional);

  std::vector<OutputSpec> outputs;
  return DataProcessorSpec{
    "zdc-raw-parser",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"log-level", o2::framework::VariantType::Int, 0, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
