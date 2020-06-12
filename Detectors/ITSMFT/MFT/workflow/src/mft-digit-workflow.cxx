// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MFTWorkflow/DigitWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // write the configuration
  o2::conf::ConfigurableParam::writeINI("o2mftdigitwriter_configuration.ini");

  return std::move(o2::mft::digit_workflow::getWorkflow());
}
