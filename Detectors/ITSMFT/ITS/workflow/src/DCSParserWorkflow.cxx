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

/// @file   DCSParserWorkflow.cxx

#include "ITSWorkflow/DCSParserWorkflow.h"
#include "ITSWorkflow/DCSParserSpec.h"

namespace o2
{
namespace its
{
namespace dcs_parser_workflow
{
framework::WorkflowSpec getWorkflow()
{
  framework::WorkflowSpec specs;

  specs.emplace_back(o2::its::getITSDCSParserSpec());

  return specs;
}
} // namespace dcs_parser_workflow
} // namespace its
} // namespace o2
