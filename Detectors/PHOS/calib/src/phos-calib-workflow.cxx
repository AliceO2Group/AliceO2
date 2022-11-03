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

#include "PHOSCalibWorkflow/PHOSPedestalCalibDevice.h"
#include "PHOSCalibWorkflow/PHOSHGLGRatioCalibDevice.h"
#include "PHOSCalibWorkflow/PHOSEnergyCalibDevice.h"
#include "PHOSCalibWorkflow/PHOSTurnonCalibDevice.h"
#include "PHOSCalibWorkflow/PHOSRunbyrunCalibDevice.h"
#include "PHOSCalibWorkflow/PHOSL1phaseCalibDevice.h"
#include "PHOSCalibWorkflow/PHOSBadMapCalibDevice.h"
#include "Framework/DataProcessorSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

// // we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  // which method should be called
  workflowOptions.push_back(ConfigParamSpec{"pedestals", o2::framework::VariantType::Bool, false, {"do pedestal calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"hglgratio", o2::framework::VariantType::Bool, false, {"do HG/LG ratio calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"turnon", o2::framework::VariantType::Bool, false, {"scan trigger turn-on curves"}});
  workflowOptions.push_back(ConfigParamSpec{"runbyrun", o2::framework::VariantType::Bool, false, {"do run by run correction calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"energy", o2::framework::VariantType::Bool, false, {"collect tree for E calib"}});
  workflowOptions.push_back(ConfigParamSpec{"badmap", o2::framework::VariantType::Bool, false, {"do bad map calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"l1phase", o2::framework::VariantType::Bool, false, {"do L1phase calculation"}});

  workflowOptions.push_back(ConfigParamSpec{"phoscalib-output-dir", o2::framework::VariantType::String, "./", {"ROOT files output directory"}});
  workflowOptions.push_back(ConfigParamSpec{"phoscalib-meta-output-dir", o2::framework::VariantType::String, "/dev/null", {"metafile output directory"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}});

  workflowOptions.push_back(ConfigParamSpec{"not-use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb phos calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"forceupdate", o2::framework::VariantType::Bool, false, {"update ccdb even difference to previous object large"}});

  // BadMap
  workflowOptions.push_back(ConfigParamSpec{"mode", o2::framework::VariantType::Int, 0, {"operation mode: 0: occupancy, 1: chi2, 2: pedestals"}});

  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

// ------------------------------------------------------------------
// we need to add workflow options before including Framework/runDataProcessing
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto doPedestals = configcontext.options().get<bool>("pedestals");
  auto doHgLgRatio = configcontext.options().get<bool>("hglgratio");
  auto doTurnOn = configcontext.options().get<bool>("turnon");
  auto doRunbyrun = configcontext.options().get<bool>("runbyrun");
  auto doEnergy = configcontext.options().get<bool>("energy");
  auto doBadMap = configcontext.options().get<bool>("badmap");
  auto doL1phase = configcontext.options().get<bool>("l1phase");
  auto useCCDB = !configcontext.options().get<bool>("not-use-ccdb");
  auto forceUpdate = configcontext.options().get<bool>("forceupdate");

  std::string outputDir = configcontext.options().get<std::string>("phoscalib-output-dir");
  if (outputDir.compare("/dev/null")) {
    outputDir = o2::utils::Str::rectifyDirectory(outputDir);
  }
  std::string metaFileDir = configcontext.options().get<std::string>("phoscalib-meta-output-dir");
  if (metaFileDir.compare("/dev/null")) {
    metaFileDir = o2::utils::Str::rectifyDirectory(metaFileDir);
  }
  bool writeRootOutput = !configcontext.options().get<bool>("disable-root-output");

  if (doPedestals && doHgLgRatio) {
    LOG(fatal) << "Can not run pedestal and HG/LG calibration simulteneously";
  }

  LOG(info) << "PHOS Calibration workflow: options";
  LOG(info) << "useCCDB = " << useCCDB;
  if (doPedestals) {
    LOG(info) << "pedestals ";
    specs.emplace_back(o2::phos::getPedestalCalibSpec(useCCDB, forceUpdate));
  } else {
    if (doHgLgRatio) {
      LOG(info) << "hglgratio ";
      specs.emplace_back(o2::phos::getHGLGRatioCalibSpec(useCCDB, forceUpdate));
    }
  }
  if (doEnergy) {
    specs.emplace_back(o2::phos::getPHOSEnergyCalibDeviceSpec(useCCDB, outputDir, metaFileDir, writeRootOutput));
  }
  if (doTurnOn) {
    LOG(info) << "TurnOn curves calculation";
    specs.emplace_back(o2::phos::getPHOSTurnonCalibDeviceSpec(useCCDB));
  }
  if (doRunbyrun) {
    LOG(info) << "Run by run correction calculation on ";
    specs.emplace_back(o2::phos::getPHOSRunbyrunCalibDeviceSpec(useCCDB, outputDir, metaFileDir, writeRootOutput));
  }
  if (doBadMap) {
    LOG(info) << "bad map calculation ";
    int mode = configcontext.options().get<int>("mode");
    specs.emplace_back(o2::phos::getBadMapCalibSpec(mode));
  }
  if (doL1phase) {
    LOG(info) << "L1phase corrections calculation on ";
    specs.emplace_back(o2::phos::getPHOSL1phaseCalibDeviceSpec());
  }
  return specs;
}
