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

#include "MCHIO/DigitWriterSpec.h"

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>

namespace o2
{
namespace mch
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getMCHDigitWriterSpec(bool mctruth)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("MCHDigitWriter",
                                "mchdigits.root",
                                "o2sim",
                                1, // default number of events
                                BranchDefinition<std::vector<o2::mch::Digit>>{InputSpec{"mchdigits", "MCH", "DIGITS"}, "MCHDigit"},
                                BranchDefinition<std::vector<o2::mch::ROFRecord>>{InputSpec{"mchrofrecords", "MCH", "DIGITROFS"}, "MCHROFRecords"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{InputSpec{"mchdigitlabels", "MCH", "DIGITSLABELS"}, "MCHMCLabels", mctruth ? 1 : 0}
                                // add more branch definitions (for example Monte Carlo labels here)
                                )();
}

o2::framework::DataProcessorSpec getDigitWriterSpec(
  bool useMC,
  std::string_view specName,
  std::string_view outfile,
  std::string_view inputDigitDataDescription,
  std::string_view inputDigitRofDataDescription)
{
  std::string input =
    fmt::format("digits:MCH/{};rofs:MCH/{}",
                inputDigitDataDescription, inputDigitRofDataDescription);

  framework::Inputs inputs{framework::select(input.c_str())};
  auto rofs = std::find_if(inputs.begin(), inputs.end(), [](const framework::InputSpec& is) { return is.binding == "rofs"; });
  auto digits = std::find_if(inputs.begin(), inputs.end(), [](const framework::InputSpec& is) { return is.binding == "digits"; });
  return framework::MakeRootTreeWriterSpec(
    std::string(specName).c_str(),
    std::string(outfile).c_str(),
    framework::MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree MCH Digits"},
    BranchDefinition<std::vector<ROFRecord>>{framework::InputSpec{*rofs}, "MCHROFRecords"},
    BranchDefinition<std::vector<Digit>>{framework::InputSpec{*digits}, "MCHDigit"},
    BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{
      framework::InputSpec{"mchdigitlabels", "MCH", "DIGITSLABELS"}, "MCHMCLabels", useMC ? 1 : 0})();
}

} // end namespace mch
} // end namespace o2
