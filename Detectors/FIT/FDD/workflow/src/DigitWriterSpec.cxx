// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitWriterSpec.cxx

#include <vector>

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DataFormatsFDD/Digit.h"
#include "FDDWorkflow/DigitWriterSpec.h"
using namespace o2::framework;

namespace o2
{
namespace fdd
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
DataProcessorSpec getFDDDigitWriterSpec()
{
  using DigitType = std::vector<o2::fdd::Digit>;
  using ChanDataType = std::vector<o2::fdd::ChannelData>;
  // Spectators for logging
  auto logger = [](DigitType const& digits) {
    LOG(INFO) << "FDDDigitWriter pulled " << digits.size() << " digits";
  };
  return MakeRootTreeWriterSpec(
    "fdd-digit-writer", "o2digit_fdd.root", "o2sim",
    BranchDefinition<DigitType>{InputSpec{"digits", "FDD", "DIGITSBC", 0},
                                "FDDDIGITSBC", "fdd-digits-branch-name", 1,
                                logger},
    BranchDefinition<ChanDataType>{InputSpec{"digch", "FDD", "DIGITSCH", 0},
                                   "FDDDIGITSCH", "fdd-chhdata-branch-name"})();
}

} // namespace fdd
} // namespace o2
