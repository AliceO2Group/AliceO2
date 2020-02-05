// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_CPVDIGITWRITER_H_
#define STEER_DIGITIZERWORKFLOW_CPVDIGITWRITER_H_

#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace cpv
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getCPVDigitWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("CPVDigitWriter",
                                "cpvdigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::cpv::Digit>>{InputSpec{"cpvdigits", "CPV", "DIGITS"}, "CPVDigit"},
                                BranchDefinition<std::vector<o2::cpv::TriggerRecord>>{InputSpec{"cpvdigitstrigrec", "CPV", "DIGITTRIGREC"}, "CPVDigitTrigRecords"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{InputSpec{"cpvdigitsmc", "CPV", "DIGITSMCTR"}, "CPVDigitMCTruth"})();
}

} // namespace cpv
} // namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_CPVDIGITWRITERSPEC_H */
