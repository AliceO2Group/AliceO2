// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file tracks-sampler-workflow.cxx
/// \brief Implementation of a DPL device to send tracks and attached clusters read from a binary file
///
/// \author Philippe Pillot, Subatech

#include <vector>

#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{"forTrackFitter", VariantType::Bool, false, {"message description as expected by TrackFitterSpec"}});
}

#include "Framework/runDataProcessing.h"

#include "TrackSamplerSpec.h"

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  bool forTrackFitter = config.options().get<bool>("forTrackFitter");
  return WorkflowSpec{o2::mch::getTrackSamplerSpec(forTrackFitter)};
}
