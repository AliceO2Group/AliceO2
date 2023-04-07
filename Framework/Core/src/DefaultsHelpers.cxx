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

#include "Framework/DefaultsHelpers.h"
#include "Framework/DataTakingContext.h"
#include <cstdlib>

namespace o2::framework
{

unsigned int DefaultsHelpers::pipelineLength()
{
  static bool override = getenv("DPL_DEFAULT_PIPELINE_LENGTH");
  if (override) {
    static unsigned int retval = atoi(getenv("DPL_DEFAULT_PIPELINE_LENGTH"));
    return retval;
  }
  DeploymentMode deploymentMode = DefaultsHelpers::deploymentMode();
  // just some reasonable numers
  // The number should really be tuned at runtime for each processor.
  if (deploymentMode == DeploymentMode::OnlineDDS || deploymentMode == DeploymentMode::OnlineECS || deploymentMode == DeploymentMode::FST) {
    return 256;
  } else {
    return 64;
  }
}

DeploymentMode DefaultsHelpers::deploymentMode()
{
  static DeploymentMode retVal = getenv("DDS_SESSION_ID") != nullptr ? DeploymentMode::OnlineDDS : (getenv("OCC_CONTROL_PORT") != nullptr ? DeploymentMode::OnlineECS : (getenv("ALIEN_JOB_ID") != nullptr ? DeploymentMode::Grid : (getenv("ALICE_O2_FST") ? DeploymentMode::FST : (DeploymentMode::Local))));
  return retVal;
}

} // namespace o2::framework
