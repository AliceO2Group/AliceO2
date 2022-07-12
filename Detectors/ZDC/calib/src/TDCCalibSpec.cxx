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

/// @file   TDCalibSpec.cxx
/// @brief  cxx file associated to TDCCalibSpec.h
/// @author luca.quaglia@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCCalib/TDCCalibConfig.h"
#include "ZDCCalib/TDCCalibSpec.h"
#include "ZDCCalib/TDCCalibData.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

TDCCalibSpec::TDCCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

TDCCalibSpec::TDCCalibSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void TDCCalibSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
  mTimer.Start(false);
}

void TDCCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ZDCTDCParam*>("tdccalib");
  pc.inputs().get<o2::zdc::TDCCalibConfig*>("tdccalibconfig");
}

void TDCCalibSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "TDCCALIB", 0)) {
    // TDC calibration object
    auto* config = (const o2::zdc::ZDCTDCParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTDCParam(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "TDCCALIBCONFIG", 0)) {
    // TDC calibration configuration
    auto* config = (const o2::zdc::TDCCalibConfig*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTDCCalibConfig(config);
  }
}

void TDCCalibSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }

  auto data = pc.inputs().get<TDCCalibData>("tdccalibdata");
  mWorker.process(data);
  for (int ih = 0; ih < TDCCalibData::NTDC; ih++) {
    o2::dataformats::FlatHisto1D<float> histoView(pc.inputs().get<gsl::span<float>>(fmt::format("tdc_1dh{}", ih).data()));
    mWorker.add(ih, histoView);
  }
}

void TDCCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  sendOutput(ec.outputs());
  LOGF(info, "ZDC TDC calibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1); //added by me
}

//________________________________________________________________
void TDCCalibSpec::sendOutput(o2::framework::DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payload = mWorker.getTDCParamUpd(); //new
  auto& info = mWorker.getCcdbObjectInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage<ZDCTDCParam>(&payload, &info); //new
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  if (mVerbosity > DbgMinimal) {
    payload.print();
  }
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_TDCcalib", 0}, *image.get()); // vector<char>
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_TDCcalib", 0}, info);         // root-serlized
  // TODO: reset the outputs once they are already sent (is it necessary?)
  // mWorker.init();
}

framework::DataProcessorSpec getTDCCalibSpec()
{
  using device = o2::zdc::TDCCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("tdccalibconfig", "ZDC", "TDCCALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTDCCalibConfig.data())));
  inputs.emplace_back("tdccalib", "ZDC", "TDCCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTDCCalib.data())));
  inputs.emplace_back("tdccalibdata", "ZDC", "TDCCALIBDATA", 0, Lifetime::Timeframe);

  char outputa[o2::header::gSizeDataDescriptionString];
  char outputd[o2::header::gSizeDataDescriptionString];
  for (int ih = 0; ih < TDCCalibData::NTDC; ih++) {
    snprintf(outputa, o2::header::gSizeDataDescriptionString, "tdc_1dh%d", ih);
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "TDC_1DH%d", ih);
    inputs.emplace_back(outputa, "ZDC", outputd, 0, Lifetime::Timeframe);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_TDCcalib"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_TDCcalib"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-tdc-calib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 1, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2