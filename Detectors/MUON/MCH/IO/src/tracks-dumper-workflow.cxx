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

#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/ConfigParamSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TrackDumperSpec.h"
#include <fmt/format.h>
#include <gsl/span>
#include <iostream>
#include <vector>

void dump(std::ostream& os, const o2::mch::TrackMCH& t)
{
  auto pt = std::sqrt(t.getPx() * t.getPx() + t.getPy() * t.getPy());
  os << fmt::format("({:s}) p {:7.2f} pt {:7.2f} nclusters: {} \n", t.getSign() == -1 ? "-" : "+", t.getP(), pt, t.getNClusters());
}

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"enable-mc", VariantType::Bool, false, ConfigParamSpec::HelpString{"Read MC info"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  bool useMC = config.options().get<bool>("enable-mc");
  WorkflowSpec specs{o2::mch::getTrackDumperSpec(useMC, "mch-tracks-dumper")};
  o2::raw::HBFUtilsInitializer hbfIni(config, specs);
  return specs;
}
