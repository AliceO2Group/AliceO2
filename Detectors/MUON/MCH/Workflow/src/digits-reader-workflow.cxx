// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file digits-reader-workflow.cxx
/// \brief Implementation of a DPL device to read digits from a binary file
///
/// \author Philippe Pillot, Subatech
/// \author Andrea Ferrero, CEA

#include "Framework/runDataProcessing.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DigitSamplerSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  WorkflowSpec wf{o2::mch::getDigitSamplerSpec()};

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, wf);

  return std::move(wf);
}
