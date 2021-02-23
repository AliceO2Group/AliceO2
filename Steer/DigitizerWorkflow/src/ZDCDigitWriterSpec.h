// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_SRC_ZDCDIGITWRITERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_ZDCDIGITWRITERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/PedestalData.h"
#include "DataFormatsZDC/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace zdc
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getZDCDigitWriterSpec(bool mctruth = true)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec("ZDCDigitWriter",
                                "zdcdigits.root",
                                "o2sim",
                                1,
                                BranchDefinition<std::vector<o2::zdc::BCData>>{InputSpec{"digitBCinput", "ZDC", "DIGITSBC"}, "ZDCDigitBC"},
                                BranchDefinition<std::vector<o2::zdc::ChannelData>>{InputSpec{"digitChinput", "ZDC", "DIGITSCH"}, "ZDCDigitCh"},
                                BranchDefinition<std::vector<o2::zdc::PedestalData>>{InputSpec{"digitPDinput", "ZDC", "DIGITSPD"}, "ZDCDigitPed"},
                                BranchDefinition<o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>>{InputSpec{"labelinput", "ZDC", "DIGITSLBL"}, "ZDCDigitLabels", mctruth ? 1 : 0})();
}

} // end namespace zdc
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_ZDCDIGITWRITERSPEC_H_ */
