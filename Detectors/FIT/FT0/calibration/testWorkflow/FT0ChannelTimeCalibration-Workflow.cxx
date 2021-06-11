// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FT0ChannelTimeCalibrationSpec.h"

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  //probably some option will be added
}

#include "Framework/runDataProcessing.h"

using namespace o2::framework;
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  workflow.emplace_back(o2::ft0::getFT0ChannelTimeCalibrationSpec());
  return workflow;
}
