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

/// \file clusters-to-tracks-workflow.cxx
/// \brief Implementation of a DPL device to run the track finder algorithm
///
/// \author Philippe Pillot, Subatech

#include "CommonUtils/ConfigurableParam.h"
#include "MCHTracking/TrackFinderSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back("configKeyValues", VariantType::String, "",
                               ConfigParamSpec::HelpString{"Semicolon separated key=value strings"});
  workflowOptions.emplace_back("disable-time-computation", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"disable track time computation from associated digits"});
  workflowOptions.emplace_back("digits", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"Send associated digits"});
  workflowOptions.emplace_back("disable-magfield-from-ccdb", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"do not read magnetic field from ccdb"});
  workflowOptions.emplace_back("original", VariantType::Bool, false,
                               ConfigParamSpec::HelpString{"use the original track finder algorithm"});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  bool computeTime = !configcontext.options().get<bool>("disable-time-computation");
  bool digits = configcontext.options().get<bool>("digits");
  bool disableCCDBMagField = configcontext.options().get<bool>("disable-magfield-from-ccdb");
  bool original = configcontext.options().get<bool>("original");
  return WorkflowSpec{o2::mch::getTrackFinderSpec(original ? "mch-track-finder-original" : "mch-track-finder",
                                                  computeTime, digits, disableCCDBMagField, original)};
}
