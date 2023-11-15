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

/// \file errors-merger-workflow.cxx
/// \brief merge the processing errors in one single output
///
/// \author Philippe Pillot, Subatech

#include <string>
#include <vector>
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "ErrorMergerSpec.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back("disable-preclustering-errors", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"Do not consider preclustering errors"});
  workflowOptions.emplace_back("disable-clustering-errors", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"Do not consider clustering errors"});
  workflowOptions.emplace_back("disable-tracking-errors", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"Do not consider tracking errors"});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  bool preclustering = !config.options().get<bool>("disable-preclustering-errors");
  bool clustering = !config.options().get<bool>("disable-clustering-errors");
  bool tracking = !config.options().get<bool>("disable-tracking-errors");
  return WorkflowSpec{o2::mch::getErrorMergerSpec("mch-error-merger", preclustering, clustering, tracking)};
}
