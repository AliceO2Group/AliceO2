// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_FT0DIGITWRITER_H_
#define STEER_DIGITIZERWORKFLOW_FT0DIGITWRITER_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace ft0
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getFT0DigitWriterSpec(bool mctruth)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("FT0DigitWriter",
                                "ft0digits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::ft0::Digit>>{InputSpec{"digitBCinput", "FT0", "DIGITSBC"}, "FT0DIGITSBC"},
                                BranchDefinition<std::vector<o2::ft0::ChannelData>>{InputSpec{"digitChinput", "FT0", "DIGITSCH"}, "FT0DIGITSCH"},
                                BranchDefinition<std::vector<o2::ft0::DetTrigInput>>{InputSpec{"digitTrinput", "FT0", "TRIGGERINPUT"}, "TRIGGERINPUT"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>>{InputSpec{"labelinput", "FT0", "DIGITSMCTR"}, "FT0DIGITSMCTR", mctruth ? 1 : 0})();
}

} // namespace ft0
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_FT0DIGITWRITER_H_ */
