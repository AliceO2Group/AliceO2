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

/// \file   MID/Workflow/src/decoded-digits-writer-workflow.cxx
/// \brief  MID decoded digits writer workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   29 October 2020

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec>
    options{
      {"mid-digits-output-filename", VariantType::String, "mid-digits-decoded.root", {"Decoded digits output file"}},
      {"mid-digits-tree-name", VariantType::String, "middigits", {"Name of tree in digits file"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MID|mid).*[W,w]riter.*"));
}

#include "Framework/runDataProcessing.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto outputFilename = cfgc.options().get<std::string>("mid-digits-output-filename");
  auto treeFilename = cfgc.options().get<std::string>("mid-digits-tree-name");
  WorkflowSpec specs;
  specs.emplace_back(MakeRootTreeWriterSpec("MIDDigitWriter",
                                            outputFilename.c_str(),
                                            treeFilename.c_str(),
                                            -1,
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ColumnData>>{InputSpec{"mid_data", o2::header::gDataOriginMID, "DATA", 0}, "MIDDigit"},
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ColumnData>>{InputSpec{"mid_data_1", o2::header::gDataOriginMID, "DATA", 1}, "MIDNoise"},
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ColumnData>>{InputSpec{"mid_data_2", o2::header::gDataOriginMID, "DATA", 2}, "MIDDead"},
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_data_rof", o2::header::gDataOriginMID, "DATAROF", 0}, "MIDROFRecords"},
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_data_rof_1", o2::header::gDataOriginMID, "DATAROF", 1}, "MIDROFRecordsNoise"},
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<o2::mid::ROFRecord>>{InputSpec{"mid_data_rof_2", o2::header::gDataOriginMID, "DATAROF", 2}, "MIDROFRecordsDead"})());

  return specs;
}
