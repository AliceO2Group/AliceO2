// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  workflowOptions.push_back(ConfigParamSpec{"ccdbpath", o2::framework::VariantType::String, "http://ccdb-test.cern.ch:8080", {"CCDB address to get current objects"}});
  workflowOptions.push_back(ConfigParamSpec{"digitspath", o2::framework::VariantType::String, "./CalibDigits.root", {"path and name of file to store calib. digits"}});

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
  auto path = configcontext.options().get<std::string>("ccdbpath");
  auto dpath = configcontext.options().get<std::string>("digitspath");
  if (doPedestals && doHgLgRatio) {
    LOG(FATAL) << "Can not run pedestal and HG/LG calibration simulteneously";
  }

  LOG(INFO) << "PHOS Calibration workflow: options";
  LOG(INFO) << "useCCDB = " << useCCDB;
  if (doPedestals) {
    LOG(INFO) << "pedestals ";
    specs.emplace_back(o2::phos::getPedestalCalibSpec(useCCDB, forceUpdate, path));
  } else {
    if (doHgLgRatio) {
      LOG(INFO) << "hglgratio ";
      specs.emplace_back(o2::phos::getHGLGRatioCalibSpec(useCCDB, forceUpdate, path));
    }
  }
  if (doEnergy) {
    LOG(INFO) << "Filling tree for energy and time calibration ";
    specs.emplace_back(o2::phos::getPHOSEnergyCalibDeviceSpec(useCCDB, path, dpath));
  }
  if (doTurnOn) {
    LOG(INFO) << "TurnOn curves calculation";
    specs.emplace_back(o2::phos::getPHOSTurnonCalibDeviceSpec(useCCDB, path));
  }
  if (doRunbyrun) {
    LOG(INFO) << "Run by run correction calculation on ";
    specs.emplace_back(o2::phos::getPHOSRunbyrunCalibDeviceSpec(useCCDB, path));
  }
  if (doBadMap) {
    LOG(INFO) << "bad map calculation ";
    short m = 0;
    // specs.emplace_back(o2::phos::getBadMapCalibSpec(useCCDB,forceUpdate,path,m));
  }
  return specs;
}
