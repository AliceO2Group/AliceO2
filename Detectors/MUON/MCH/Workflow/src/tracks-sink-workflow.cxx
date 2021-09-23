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

/// \file tracks-sink-workflow.cxx
/// \brief Implementation of a DPL device to write tracks and attached clusters into a binary file
///
/// \author Philippe Pillot, Subatech

#include <vector>

#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{"mchTracksOnly", VariantType::Bool, false, {"only write MCH tracks and attached clusters"}});
  workflowOptions.push_back(
    ConfigParamSpec{"tracksAtVertexOnly", VariantType::Bool, false, {"only write track parameters at vertex"}});
}

#include "Framework/runDataProcessing.h"

#include "TrackSinkSpec.h"

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  bool mchTracks = !config.options().get<bool>("tracksAtVertexOnly");
  bool tracksAtVtx = !config.options().get<bool>("mchTracksOnly");
  return WorkflowSpec{o2::mch::getTrackSinkSpec("mch-track-sink", mchTracks, tracksAtVtx)};
}
