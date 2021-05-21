// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_SRC_MCHDIGITWRITERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_MCHDIGITWRITERSPEC_H_

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
                                1, //default number of events
                                BranchDefinition<std::vector<o2::mch::Digit>>{InputSpec{"mchdigits", "MCH", "DIGITS"}, "MCHDigit"},
                                BranchDefinition<std::vector<o2::mch::ROFRecord>>{InputSpec{"mchrofrecords", "MCH", "DIGITROFS"}, "MCHROFRecords"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{InputSpec{"mchdigitlabels", "MCH", "DIGITSLABELS"}, "MCHMCLabels", mctruth ? 1 : 0}
                                // add more branch definitions (for example Monte Carlo labels here)
                                )();
}

} // end namespace mch
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_MCHDIGITWRITERSPEC_H_ */
