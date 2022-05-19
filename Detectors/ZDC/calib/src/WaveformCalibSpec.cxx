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

/// @file   WaveformCalibSpec.cxx
/// @brief  ZDC reconstruction
/// @author pietro.cortese@cern.ch

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
#include "Framework/DataRefUtils.h"
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
#include "ZDCCalib/WaveformCalibData.h"
#include "ZDCCalib/WaveformCalibSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

WaveformCalibSpec::WaveformCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

WaveformCalibSpec::WaveformCalibSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void WaveformCalibSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
  mTimer.CpuTime();
  mTimer.Start(false);
}

void WaveformCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::WaveformCalibConfig*>("wavecalibconfig");
}

void WaveformCalibSpec::run(ProcessingContext& pc)
{
  updateTimeDependentParams(pc);
  if (!mInitialized) {
    mInitialized = true;
    std::string loadedConfFiles = "Loaded ZDC configuration files:";
    std::string ct = "WaveformCalibConfig";
    std::string cn = "wavecalibconfig";
    // WaveformCalib configuration
    auto config = pc.inputs().get<o2::zdc::WaveformCalibConfig*>(cn);
    if (!config) {
      LOG(fatal) << "Missing calibration object: " << ct;
      return;
    } else {
      loadedConfFiles += " ";
      loadedConfFiles += cn;
      if (mVerbosity > DbgZero) {
        LOG(info) << "Loaded configuration object: " << ct;
        config->print();
      }
      mWorker.setConfig(config.get());
    }
    LOG(info) << loadedConfFiles;
    mTimer.CpuTime();
    mTimer.Start(false);
  }

  auto data = pc.inputs().get<WaveformCalibData>("waveformcalibdata");
  mWorker.process(data);
}

void WaveformCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  sendOutput(ec.outputs());
  LOGF(info, "ZDC Waveformcalibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

//________________________________________________________________
void WaveformCalibSpec::sendOutput(o2::framework::DataAllocator& output)
{
  /*
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payload = mWorker.getTowerParamUpd();
  auto& info = mWorker.getCcdbObjectInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage<ZDCTowerParam>(&payload, &info);
  if (mVerbosity > DbgMinimal) {
    payload.print();
  }
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_Waveformcalib", 0}, *image.get()); // vector<char>
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_Waveformcalib", 0}, info);         // root-serialized
  // TODO: reset the outputs once they are already sent (is it necessary?)
  // mWorker.init();
  */
}

framework::DataProcessorSpec getWaveformCalibSpec()
{
  using device = o2::zdc::WaveformCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("intercalibconfig", "ZDC", "INTERCALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathWaveformCalibConfig.data())));
  inputs.emplace_back("energycalib", "ZDC", "ENERGYCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathEnergyCalib.data())));
  inputs.emplace_back("towercalib", "ZDC", "TOWERCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTowerCalib.data())));
  inputs.emplace_back("intercalibdata", "ZDC", "INTERCALIBDATA", 0, Lifetime::Timeframe);

  char outputa[o2::header::gSizeDataDescriptionString];
  char outputd[o2::header::gSizeDataDescriptionString];
  for (int ih = 0; ih < (2 * WaveformCalibData::NH); ih++) {
    snprintf(outputa, o2::header::gSizeDataDescriptionString, "inter_1dh%d", ih);
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "INTER_1DH%d", ih);
    inputs.emplace_back(outputa, "ZDC", outputd, 0, Lifetime::Timeframe);
  }
  for (int ih = 0; ih < WaveformCalibData::NH; ih++) {
    snprintf(outputa, o2::header::gSizeDataDescriptionString, "inter_2dh%d", ih);
    snprintf(outputd, o2::header::gSizeDataDescriptionString, "INTER_2DH%d", ih);
    inputs.emplace_back(outputa, "ZDC", outputd, 0, Lifetime::Timeframe);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDCWaveformcalib"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDCWaveformcalib"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-calib-towers",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::Int, 5, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 500, {"minimum number of entries to fit single time slot"}},
      {"verbosity-level", o2::framework::VariantType::Int, 1, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
