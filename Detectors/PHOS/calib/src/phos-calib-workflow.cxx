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
#include "Framework/DataProcessorSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"

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
  //
  workflowOptions.push_back(ConfigParamSpec{"not-use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb phos calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"forceupdate", o2::framework::VariantType::Bool, false, {"update ccdb even difference to previous object large"}});
  workflowOptions.push_back(ConfigParamSpec{"digitspath", o2::framework::VariantType::String, "./CalibDigits.root", {"path and name of file to store calib. digits"}});

  workflowOptions.push_back(ConfigParamSpec{"ptminmgg", o2::framework::VariantType::Float, 1.5f, {"minimal pt to fill mgg calib histos"}});
  workflowOptions.push_back(ConfigParamSpec{"eminhgtime", o2::framework::VariantType::Float, 1.5f, {"minimal E (GeV) to fill HG time calib histos"}});
  workflowOptions.push_back(ConfigParamSpec{"eminlgtime", o2::framework::VariantType::Float, 5.f, {"minimal E (GeV) to fill LG time calib histos"}});

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
  auto useCCDB = !configcontext.options().get<bool>("not-use-ccdb");
  auto forceUpdate = configcontext.options().get<bool>("forceupdate");
  auto dpath = configcontext.options().get<std::string>("digitspath");
  auto path = o2::base::NameConf::getCCDBServer();

  float ptMin = configcontext.options().get<float>("ptminmgg");
  float eMinHGTime = configcontext.options().get<float>("eminhgtime");
  float eMinLGTime = configcontext.options().get<float>("eminlgtime");

  if (doPedestals && doHgLgRatio) {
    LOG(fatal) << "Can not run pedestal and HG/LG calibration simulteneously";
  }

  LOG(info) << "PHOS Calibration workflow: options";
  LOG(info) << "useCCDB = " << useCCDB;
  if (doPedestals) {
    LOG(info) << "pedestals ";
    specs.emplace_back(o2::phos::getPedestalCalibSpec(useCCDB, forceUpdate, path));
  } else {
    if (doHgLgRatio) {
      LOG(info) << "hglgratio ";
      specs.emplace_back(o2::phos::getHGLGRatioCalibSpec(useCCDB, forceUpdate, path));
    }
  }
  if (doEnergy) {
    LOG(info) << "Filling tree for energy and time calibration ";
    specs.emplace_back(o2::phos::getPHOSEnergyCalibDeviceSpec(useCCDB, path, dpath, ptMin, eMinHGTime, eMinLGTime));
  }
  if (doTurnOn) {
    LOG(info) << "TurnOn curves calculation";
    specs.emplace_back(o2::phos::getPHOSTurnonCalibDeviceSpec(useCCDB, path));
  }
  if (doRunbyrun) {
    LOG(info) << "Run by run correction calculation on ";
    specs.emplace_back(o2::phos::getPHOSRunbyrunCalibDeviceSpec(useCCDB, path));
  }
  if (doBadMap) {
    LOG(info) << "bad map calculation ";
    short m = 0;
    // specs.emplace_back(o2::phos::getBadMapCalibSpec(useCCDB,forceUpdate,path,m));
  }
  return specs;
}
