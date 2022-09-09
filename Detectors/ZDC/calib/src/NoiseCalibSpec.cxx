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

/// @file   NoiseCalibSpec.cxx
/// @brief  ZDC baseline calibration
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include <gsl/span>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataRefUtils.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCCalib/NoiseCalibSpec.h"
#include "ZDCCalib/NoiseCalibData.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

NoiseCalibSpec::NoiseCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

NoiseCalibSpec::NoiseCalibSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void NoiseCalibSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
}

void NoiseCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  //pc.inputs().get<o2::zdc::NoiseCalibConfig*>("calibconfig");
}

void NoiseCalibSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
//   if (matcher == ConcreteDataMatcher("ZDC", "NOISECALIBCONFIG", 0)) {
//     mWorker.setConfig((const o2::zdc::NoiseCalibConfig*)obj);
//   }
}

void NoiseCalibSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }
  auto data = pc.inputs().get<o2::zdc::NoiseCalibSummaryData*>("basecalibdata");
  mWorker.process(data.get());
}

void NoiseCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  sendOutput(ec.outputs());
  LOGF(info, "ZDC Noise calibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

//________________________________________________________________
void NoiseCalibSpec::sendOutput(o2::framework::DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payload = mWorker.getParamUpd();
  auto& info = mWorker.getCcdbObjectInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage<NoiseParam>(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  if (mVerbosity > DbgMinimal) {
    payload.print();
  }
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDCNoisecalib", 0}, *image.get()); // vector<char>
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDCNoisecalib", 0}, info);         // root-serialized
  // TODO: reset the outputs once they are already sent (is it necessary?)
  // mWorker.init();
}

framework::DataProcessorSpec getNoiseCalibSpec()
{
  using device = o2::zdc::NoiseCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("noisecalibdata", "ZDC", "NOISECALIBDATA", 0, Lifetime::Timeframe);
//  inputs.emplace_back("calibconfig", "ZDC", "BASECALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathNoiseCalibConfig.data()));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDCNoisecalib"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDCNoisecalib"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-calib-noise",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 1, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
