// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterWriterSpec.cxx

#include <vector>
#include "DataFormatsCTP/Digits.h"
#include "CTPWorkflow/WriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace ctp
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using DigitType = std::vector<o2::ctp::CTPDigit>;
using MCLabelType = o2::dataformats::MCTruthContainer<MCCompLabel>;
using namespace o2::header;


DataProcessorSpec getDigitWriterSpec(bool useMC)
{
  // Spectators for logging
  // this is only to restore the original behavior
  auto DigitsSize = std::make_shared<int>(0);
  auto DigitsSizeGetter = [DigitsSize](DigitType const& Digits) {
    *DigitsSize = Digits.size();
  };

  if (useMC) {
    return MakeRootTreeWriterSpec("ctp-digit-writer",
                                  "ctpdigits.root",
                                  MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CTP digits"},
                                  BranchDefinition<DigitType>{InputSpec{"CTPDigit", "CTP", "DIGITS", 0},
                                                              "CTPDigit", DigitsSizeGetter},
                                  BranchDefinition<MCLabelType>{InputSpec{"clusMC", "CPV", "DIGITSMCTR", 0},
                                                                "CPVDigitMCTruth"})();
  } else {
    return MakeRootTreeWriterSpec("ctp-digit-writer",
                                  "ctpdigits.root",
                                  MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CTP digits"},
                                  BranchDefinition<DigitType>{InputSpec{"CTPDigit", "CTP", "DIGITS", 0},
                                                              "CTPDigit", DigitsSizeGetter})();
  }
}

} // namespace cpv
} // namespace o2
