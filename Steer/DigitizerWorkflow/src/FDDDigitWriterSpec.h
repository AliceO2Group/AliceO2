// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_SRC_FDDDIGITWRITERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_FDDDIGITWRITERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace fdd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getFDDDigitWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("FDDDigitWriter",
                                "fdddigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::fdd::Digit>>{InputSpec{"digitinput", "FDD", "DIGITS"}, "FDDDigit"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>>{InputSpec{"labelinput", "FDD", "DIGITSMC"}, "FDDDigitLabels"})();
}

} // end namespace fdd
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_FDDDIGITWRITERSPEC_H_ */
