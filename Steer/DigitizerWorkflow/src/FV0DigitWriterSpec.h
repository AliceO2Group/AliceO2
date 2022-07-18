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

#ifndef STEER_DIGITIZERWORKFLOW_FV0DIGITWRITER_H_
#define STEER_DIGITIZERWORKFLOW_FV0DIGITWRITER_H_

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/Digit.h"
#include "DataFormatsFV0/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace fv0
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getFV0DigitWriterSpec(bool mctruth = true)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("FV0DigitWriter",
                                "fv0digits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::fv0::Digit>>{InputSpec{"fv0digitBCinput", "FV0", "DIGITSBC"}, "FV0DigitBC"},
                                BranchDefinition<std::vector<o2::fv0::ChannelData>>{InputSpec{"fv0digitChinput", "FV0", "DIGITSCH"}, "FV0DigitCh"},
                                BranchDefinition<std::vector<o2::fv0::DetTrigInput>>{InputSpec{"fv0digitTrinput", "FV0", "TRIGGERINPUT"}, "TRIGGERINPUT"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>>{InputSpec{"fv0labelinput", "FV0", "DIGITLBL"}, "FV0DigitLabels", mctruth ? 1 : 0})();
}

} // namespace fv0
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_FV0DIGITWRITER_H_ */
