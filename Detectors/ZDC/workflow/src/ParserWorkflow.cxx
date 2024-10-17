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

/// @file   ParserWorkflow.cxx

#include "ZDCWorkflow/ParserWorkflow.h"
#include "ZDCWorkflow/DigitReaderSpec.h"

namespace o2
{
namespace zdc
{

framework::WorkflowSpec getRecoWorkflow(const int verbosity)
{
  framework::WorkflowSpec specs;
  specs.emplace_back(o2::zdc::getDigitReaderSpec(false));
//  specs.emplace_back(o2::zdc::getDigitRecoSpec(verbosity, enableDebugOut, enableZDCTDCCorr, enableZDCEnergyParam, enableZDCTowerParam, enableBaselineParam));
  return specs;
}

} // namespace zdc
} // namespace o2
