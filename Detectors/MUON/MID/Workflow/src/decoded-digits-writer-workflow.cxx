// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/decoded-digits-writer-workflow.cxx
/// \brief  MID decoded digits writer workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   29 October 2020

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec>
    options{
      {"output-filename", VariantType::String, "mid-digits-decoded.root", {"Decoded digits output file"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto outputFilename = cfgc.options().get<std::string>("output-filename");
  WorkflowSpec specs;
  specs.emplace_back(MakeRootTreeWriterSpec("MIDDigitWriter",
                                            outputFilename.c_str(),
                                            "middigits",
                                            -1,
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ColumnData>>{InputSpec{"mid_data", o2::header::gDataOriginMID, "DATA"}, "MIDDigit"},
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_data_rof", o2::header::gDataOriginMID, "DATAROF"}, "MIDDigitROF"})());

  return specs;
}
