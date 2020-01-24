// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_PHOSDIGITWRITER_H_
#define STEER_DIGITIZERWORKFLOW_PHOSDIGITWRITER_H_

#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsPHOS/MCLabel.h"

namespace o2
{
namespace phos
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getPHOSDigitWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("PHOSDigitWriter",
                                "phosdigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::phos::Digit>>{InputSpec{"phosdigits", "PHS", "DIGITS"}, "PHOSDigit"},
                                BranchDefinition<std::vector<o2::phos::TriggerRecord>>{InputSpec{"phosdigitstrigrec", "PHS", "DIGITTRIGREC"}, "PHOSDigitTrigRecords"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::phos::MCLabel>>{InputSpec{"phosdigitsmc", "PHS", "DIGITSMCTR"}, "PHOSDigitMCTruth"})();
}

} // namespace phos
} // namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_PHOSDIGITWRITERSPEC_H */
