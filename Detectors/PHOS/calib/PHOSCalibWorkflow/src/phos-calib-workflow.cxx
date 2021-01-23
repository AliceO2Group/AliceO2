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
#include "PHOSCalibWorkflow/PHOSCalibCollector.h"
#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

// // we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb phos calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"forceupdate", o2::framework::VariantType::Bool, false, {"update ccdb even difference to previous object large"}});
  workflowOptions.push_back(ConfigParamSpec{"pedestals", o2::framework::VariantType::Bool, false, {"do pedestal calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"hglgratio", o2::framework::VariantType::Bool, false, {"do HG/LG ratio calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"etree", o2::framework::VariantType::Int, -1, {"collect tree for E calib (0: data scan, 1: iteration, 2: CalibParam calculaion)"}});
  workflowOptions.push_back(ConfigParamSpec{"badmap", o2::framework::VariantType::Bool, false, {"do bad map calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"path", o2::framework::VariantType::String, "./", {"path to store temp files"}});
}

// ------------------------------------------------------------------
// we need to add workflow options before including Framework/runDataProcessing
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  auto useCCDB = configcontext.options().get<bool>("use-ccdb");
  auto forceUpdate = configcontext.options().get<bool>("forceupdate");
  auto doPedestals = configcontext.options().get<bool>("pedestals");
  auto doHgLgRatio = configcontext.options().get<bool>("hglgratio");
  auto doEtree = configcontext.options().get<int>("etree");
  auto doBadMap = configcontext.options().get<bool>("badmap");
  auto path = configcontext.options().get<std::string>("path");
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
  if (doBadMap) {
    LOG(INFO) << "bad map calculation ";
    short m = 0;
    // specs.emplace_back(o2::phos::getBadMapCalibSpec(useCCDB,forceUpdate,path,m));
  }
  if (doEtree >= 0) {
    LOG(INFO) << "Filling tree for energy and time calibration ";
    specs.emplace_back(o2::phos::getPHOSCalibCollectorDeviceSpec(doEtree));
  }
  return specs;
}
