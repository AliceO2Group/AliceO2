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

#ifndef STEER_DIGITIZERWORKFLOW_SRC_HMPDIGITWRITERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_HMPDIGITWRITERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace hmpid
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getHMPIDDigitWriterSpec(bool mctruth = true)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("HMPDigitWriter",
                                "hmpiddigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::hmpid::Digit>>{InputSpec{"hmpdigitinput", "HMP", "DIGITS"}, "HMPDigit"},
                                BranchDefinition<std::vector<o2::hmpid::Trigger>>{InputSpec{"hmpinteractionrecords", "HMP", "INTRECORDS"}, "InteractionRecords"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>{InputSpec{"hmplabelinput", "HMP", "DIGITLBL"}, "HMPDigitLabels", mctruth ? 1 : 0})();
}

} // end namespace hmpid
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_HMPIDDIGITWRITERSPEC_H_ */
