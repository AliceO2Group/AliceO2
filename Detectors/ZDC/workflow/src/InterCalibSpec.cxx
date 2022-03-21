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

/// @file   InterCalibSpec.cxx
/// @brief  ZDC reconstruction
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/InterCalib.h"
#include "ZDCCalib/InterCalibConfig.h"
#include "ZDCWorkflow/InterCalibSpec.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

InterCalibSpec::InterCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

InterCalibSpec::InterCalibSpec(const int verbosity)
  : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void InterCalibSpec::init(o2::framework::InitContext& ic)
{
  mccdbHost = ic.options().get<std::string>("ccdb-url");
}

void InterCalibSpec::run(ProcessingContext& pc)
{
  InterCalib work;
  if (!mInitialized) {
    mInitialized = true;
    // Initialization from CCDB
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL(mccdbHost);
    std::string loadedConfFiles = "Loaded ZDC configuration files for timestamp " + std::to_string(mgr.getTimestamp()) + ":";

    // Energy calibration
    auto* energyParam = mgr.get<o2::zdc::ZDCEnergyParam>(o2::zdc::CCDBPathEnergyCalib);
    if (!energyParam) {
      LOG(fatal) << "Missing ZDCEnergyParam calibration object";
    } else {
      loadedConfFiles += " ZDCEnergyParam";
      if (mVerbosity > DbgZero) {
        LOG(info) << "Loaded Energy calibration ZDCEnergyParam for timestamp " << mgr.getTimestamp();
        energyParam->print();
      }
    }

    // Tower calibration
    auto* towerParam = mgr.get<o2::zdc::ZDCTowerParam>(o2::zdc::CCDBPathTowerCalib);
    if (!towerParam) {
      LOG(fatal) << "Missing ZDCTowerParam calibration object";
    } else {
      loadedConfFiles += " ZDCTowerParam";
      if (mVerbosity > DbgZero) {
        LOG(info) << "Loaded Tower calibration ZDCTowerParam for timestamp " << mgr.getTimestamp();
        towerParam->print();
      }
    }

    // InterCalib configuration
    auto* interConfig = mgr.get<o2::zdc::InterCalibConfig>(o2::zdc::CCDBPathInterCalibConfig);
    if (!interConfig) {
      LOG(fatal) << "Missing InterCalibConfig calibration object";
    } else {
      loadedConfFiles += " InterCalibConfig";
      if (mVerbosity > DbgZero) {
        LOG(info) << "Loaded InterCalib configuration object for timestamp " << mgr.getTimestamp();
        interConfig->print();
      }
    }

    work.setEnergyParam(energyParam);
    work.setTowerParam(towerParam);
    work.setInterCalibConfig(interConfig);

    LOG(info) << loadedConfFiles;
  }

  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto bcrec = pc.inputs().get<gsl::span<o2::zdc::BCRecData>>("bcrec");
  auto energy = pc.inputs().get<gsl::span<o2::zdc::ZDCEnergy>>("energy");
  auto tdc = pc.inputs().get<gsl::span<o2::zdc::ZDCTDCData>>("tdc");
  auto info = pc.inputs().get<gsl::span<uint16_t>>("info");
  work.init();
  work.process(bcrec, energy, tdc, info);
  mTimer.Stop();
}

void InterCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "ZDC Intercalibration total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

framework::DataProcessorSpec getInterCalibSpec(const int verbosity = 0)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("bcrec", "ZDC", "BCREC", 0, Lifetime::Timeframe);
  inputs.emplace_back("energy", "ZDC", "ENERGY", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdc", "ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("info", "ZDC", "INFO", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "zdc-intercalib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<InterCalibSpec>(verbosity)},
    o2::framework::Options{{"ccdb-url", o2::framework::VariantType::String, o2::base::NameConf::getCCDBServer(), {"CCDB Url"}}}};
}

} // namespace zdc
} // namespace o2
