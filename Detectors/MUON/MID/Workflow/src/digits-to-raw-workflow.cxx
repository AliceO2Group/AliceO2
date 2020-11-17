// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/digits-to-raw-workflow.cxx
/// \brief  MID raw to digits workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   02 October 2019

#include <string>
#include <vector>
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Variant.h"
#include "MIDWorkflow/DigitReaderSpec.h"
#include "MIDWorkflow/RawWriterSpec.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {keyvaluehelp}});
  workflowOptions.push_back(ConfigParamSpec{"hbfutils-config", o2::framework::VariantType::String, std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE), {"config file for HBFUtils (or none), used for raw output only!!!"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  std::string confDig = configcontext.options().get<std::string>("hbfutils-config");
  if (!confDig.empty() && confDig != "none") {
    o2::conf::ConfigurableParam::updateFromFile(confDig, "HBFUtils");
  }
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  specs.emplace_back(o2::mid::getDigitReaderSpec(false));
  specs.emplace_back(o2::mid::getRawWriterSpec());

  return specs;
}
