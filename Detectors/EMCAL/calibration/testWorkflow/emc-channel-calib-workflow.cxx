// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   emc-channel-calib-workflow.cxx
/// @author Hannah Bossi
/// @since  2020-12-01
/// @brief  Basic workflow for EMCAL bad channel calibration (adapted from tof-calib-workflow.cxx)

#include "EMCALChannelCalibratorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Algorithm/RangeTokenizer.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // we don't have any configuration options for this task
}

#include "Framework/runDataProcessing.h" // the main driver

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  specs.emplace_back(getEMCALChannelCalibDeviceSpec());
  return specs;
}
