// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_SRC_TRDDIGITWRITERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_TRDDIGITWRITERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "Utils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "TRDBase/Digit.h"

namespace o2
{
namespace trd
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getTRDDigitWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("TRDDigitWriter",
                                "trddigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::trd::Digit>>{ InputSpec{ "input", "TRD", "DIGITS" }, "TRDDigit" }
                                // add more branch definitions (for example Monte Carlo labels here)
                                )();
}

} // end namespace trd
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_TRDDIGITWRITERSPEC_H_ */
