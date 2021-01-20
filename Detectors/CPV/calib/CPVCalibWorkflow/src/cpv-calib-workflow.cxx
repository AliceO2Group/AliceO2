// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CPVCalibWorkflow/CPVPedestalCalibDevice.h"
#include "CPVCalibWorkflow/CPVGainCalibDevice.h"
#include "CPVCalibWorkflow/CPVBadMapCalibDevice.h"
#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

// // we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb cpv calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"forceupdate", o2::framework::VariantType::Bool, false, {"update ccdb even difference to previous object large"}});
  workflowOptions.push_back(ConfigParamSpec{"pedestals", o2::framework::VariantType::Bool, false, {"do pedestal calculation"}});
  workflowOptions.push_back(ConfigParamSpec{"gains", o2::framework::VariantType::Bool, false, {"do gain calculation"}});
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
  auto doGain = configcontext.options().get<bool>("gains");
  auto doBadMap = configcontext.options().get<bool>("badmap");
  auto path = configcontext.options().get<std::string>("path");
  if (doPedestals && doGain) {
    LOG(FATAL) << "Can not run pedestal and gain calibration simulteneously";
  }

  LOG(INFO) << "CPV Calibration workflow: options";
  LOG(INFO) << "useCCDB = " << useCCDB;
  if (doPedestals) {
    LOG(INFO) << "pedestals ";
    specs.emplace_back(o2::cpv::getPedestalCalibSpec(useCCDB, forceUpdate, path));
  } else {
    if (doGain) {
      LOG(INFO) << "gain calculation";
      specs.emplace_back(o2::cpv::getGainCalibSpec(useCCDB, forceUpdate, path));
    }
  }
  if (doBadMap) {
    LOG(INFO) << "bad map calculation ";
    short m = 0;
    specs.emplace_back(o2::cpv::getBadMapCalibSpec(useCCDB, forceUpdate, path, m));
  }
  return specs;
}
