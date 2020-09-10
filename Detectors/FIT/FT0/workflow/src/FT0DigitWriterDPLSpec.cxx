// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0DigitWriterSpec.cxx

#include <vector>

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/Digit.h"
#include "FT0Workflow/FT0DigitWriterDPLSpec.h"
using namespace o2::framework;

namespace o2
{
namespace ft0
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
DataProcessorSpec getFT0DigitWriterDPLSpec()
{
  using DigitType = std::vector<o2::ft0::Digit>;
  using ChanDataType = std::vector<o2::ft0::ChannelData>;
  // Spectators for logging
  auto logger = [](DigitType const& digits) {
    LOG(INFO) << "FT0DigitWriter pulled " << digits.size() << " digits";
  };
  return MakeRootTreeWriterSpec(
    "ft0-digit-writer", "o2digit_ft0.root", "o2sim",
    BranchDefinition<DigitType>{InputSpec{"digits", "FT0", "DIGITSBC", 0},
                                "FT0DIGITSBC", "ft0-digits-branch-name", 1,
                                logger},
    BranchDefinition<ChanDataType>{InputSpec{"digch", "FT0", "DIGITSCH", 0},
                                   "FT0DIGITSCH", "ft0-chhdata-branch-name"})();
}

} // namespace ft0
} // namespace o2
