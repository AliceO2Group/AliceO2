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

#include "ITSWorkflow/DCSParserWorkflow.h"
#include "ITSWorkflow/DCSParserSpec.h"

using namespace o2::framework;

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  LOG(info) << "Initializing O2 ITS-DCS String Parser:";

  return std::move(o2::its::dcs_parser_workflow::getWorkflow());
}
