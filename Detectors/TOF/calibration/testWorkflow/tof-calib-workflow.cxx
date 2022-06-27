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

#include "TOFChannelCalibratorSpec.h"
#include "LHCClockCalibratorSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/CalibInfoCluster.h"
#include "Framework/ConfigParamSpec.h"
using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"do-lhc-phase", o2::framework::VariantType::Bool, false, {"do LHC clock phase calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"do-channel-offset", o2::framework::VariantType::Bool, false, {"do TOF channel offset calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"attach-channel-offset-to-lhcphase", o2::framework::VariantType::Bool, false, {"do TOF channel offset calibration using the LHCphase previously calculated in the same workflow"}});
  workflowOptions.push_back(ConfigParamSpec{"cosmics", o2::framework::VariantType::Bool, false, {"for cosmics data"}});
  workflowOptions.push_back(ConfigParamSpec{"perstrip", o2::framework::VariantType::Bool, false, {"offsets per strip"}});
  workflowOptions.push_back(ConfigParamSpec{"safe-mode", o2::framework::VariantType::Bool, false, {"require safe mode (discard strange TF)"}});
  workflowOptions.push_back(ConfigParamSpec{"follow-ccdb-updates", o2::framework::VariantType::Bool, false, {"whether to update the CCDB entries during calibration"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto useCCDB = configcontext.options().get<bool>("use-ccdb");
  auto doLHCcalib = configcontext.options().get<bool>("do-lhc-phase");
  auto doChannelOffsetCalib = configcontext.options().get<bool>("do-channel-offset");
  auto attachChannelOffsetToLHCphase = configcontext.options().get<bool>("attach-channel-offset-to-lhcphase");
  auto isCosmics = configcontext.options().get<bool>("cosmics");
  auto perstrip = configcontext.options().get<bool>("perstrip");
  auto safe = configcontext.options().get<bool>("safe-mode");
  auto followCCDBUpdates = configcontext.options().get<bool>("follow-ccdb-updates");

  if (isCosmics) {
    LOG(info) << "Cosmics set!!!! No LHC phase, Yes channel offset";
    doChannelOffsetCalib = true;
    doLHCcalib = false;
  }

  if (!doLHCcalib && attachChannelOffsetToLHCphase) {
    LOG(info) << "Over-writing attachChannelOffsetToLHCphase because we are not doing the LHCphase calibration";
    attachChannelOffsetToLHCphase = false;
  }

  LOG(info) << "TOF Calibration workflow: options";
  LOG(info) << "doLHCcalib = " << doLHCcalib;
  LOG(info) << "doChannelOffsetCalib = " << doChannelOffsetCalib;
  LOG(info) << "useCCDB = " << useCCDB;
  LOG(info) << "attachChannelOffsetToLHCphase = " << attachChannelOffsetToLHCphase;
  LOG(info) << "followCCDBUpdates = " << followCCDBUpdates;

  if (doLHCcalib) {
    specs.emplace_back(getLHCClockCalibDeviceSpec(useCCDB, followCCDBUpdates));
  }
  if (doChannelOffsetCalib) {
    if (!isCosmics) {
      specs.emplace_back(getTOFChannelCalibDeviceSpec<o2::dataformats::CalibInfoTOF>(useCCDB, followCCDBUpdates, attachChannelOffsetToLHCphase, isCosmics, perstrip, safe));
    } else {
      specs.emplace_back(getTOFChannelCalibDeviceSpec<o2::tof::CalibInfoCluster>(useCCDB, followCCDBUpdates, attachChannelOffsetToLHCphase, isCosmics));
    }
  }
  return specs;
}
