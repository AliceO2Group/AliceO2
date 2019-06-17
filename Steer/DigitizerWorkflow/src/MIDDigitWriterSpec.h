// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_SRC_MIDDIGITWRITERSPEC_H
#define STEER_DIGITIZERWORKFLOW_SRC_MIDDIGITWRITERSPEC_H

#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/MCLabel.h"

namespace o2
{
namespace mid
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getMIDDigitWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("MIDDigitWriter",
                                "middigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::mid::ColumnDataMC>>{ InputSpec{ "middigits", "MID", "DIGITS" }, "MIDDigit" },
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::mid::MCLabel>>{ InputSpec{ "middigitlabels", "MID", "DIGITSMC" }, "MIDDigitMCLabels" }
                                // add more branch definitions (for example Monte Carlo labels here)
                                )();
}

} // namespace mid
} // namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_MIDDIGITWRITERSPEC_H */
