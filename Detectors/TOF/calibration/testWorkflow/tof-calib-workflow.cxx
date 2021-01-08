// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFChannelCalibratorSpec.h"
#include "LHCClockCalibratorSpec.h"
#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"do-lhc-phase", o2::framework::VariantType::Bool, true, {"do LHC clock phase calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"do-channel-offset", o2::framework::VariantType::Bool, false, {"do TOF channel offset calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"attach-channel-offset-to-lhcphase", o2::framework::VariantType::Bool, false, {"do TOF channel offset calibration using the LHCphase previously calculated in the same workflow"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  auto useCCDB = configcontext.options().get<bool>("use-ccdb");
  auto doLHCcalib = configcontext.options().get<bool>("do-lhc-phase");
  auto doChannelOffsetCalib = configcontext.options().get<bool>("do-channel-offset");
  auto attachChannelOffsetToLHCphase = configcontext.options().get<bool>("attach-channel-offset-to-lhcphase");
  if (!doLHCcalib && attachChannelOffsetToLHCphase) {
    LOG(INFO) << "Over-writing attachChannelOffsetToLHCphase because we are not doing the LHCphase calibration";
    attachChannelOffsetToLHCphase = false;
  }

  LOG(INFO) << "TOF Calibration workflow: options";
  LOG(INFO) << "doLHCcalib = " << doLHCcalib;
  LOG(INFO) << "doChannelOffsetCalib = " << doChannelOffsetCalib;
  LOG(INFO) << "useCCDB = " << useCCDB;
  LOG(INFO) << "attachChannelOffsetToLHCphase = " << attachChannelOffsetToLHCphase;
  if (doLHCcalib) {
    specs.emplace_back(getLHCClockCalibDeviceSpec());
  }
  if (doChannelOffsetCalib) {
    specs.emplace_back(getTOFChannelCalibDeviceSpec(useCCDB, attachChannelOffsetToLHCphase));
  }
  return specs;
}
