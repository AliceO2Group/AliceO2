// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/digits-reader-workflow.cxx
/// \brief  MID digits reader workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 October 2020

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDSimulation/MCLabel.h"
#include "MIDWorkflow/DigitReaderSpec.h"
#include "MIDWorkflow/ZeroSuppressionSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"disable-mc", VariantType::Bool, false, {"Do not propagate MC info"}},
    {"disable-zero-suppression", VariantType::Bool, false, {"Do not apply zero suppression"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool disableZS = cfgc.options().get<bool>("disable-zero-suppression");
  bool useMC = !cfgc.options().get<bool>("disable-mc");

  WorkflowSpec specs;
  specs.emplace_back(o2::mid::getDigitReaderSpec(useMC, disableZS ? "DATA" : "DATAMC"));
  if (!disableZS) {
    specs.emplace_back(o2::mid::getZeroSuppressionSpec(useMC));
  }

  return specs;
}
