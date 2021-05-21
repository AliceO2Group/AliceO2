// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CTPDigitWriterSpec.cxx
/// \author Roman Lietava

#include "CTPWorkflow/CTPDigitWriterSpec.h"

namespace o2
{
namespace ctp
{
template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

framework::DataProcessorSpec getCTPDigitWriterSpec(bool raw)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  // Spectators for logging
  auto logger = [](std::vector<o2::ctp::CTPDigit> const& vecDigits) {
    LOG(INFO) << "CTPDigitWriter pulled " << vecDigits.size() << " digits";
  };
  return MakeRootTreeWriterSpec(raw ? "ctp-digit-writer-dec" : "ctp-digit-writer",
                                raw ? "o2_ctpdigits.root" : "ctpdigits.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CTP digits"},
                                BranchDefinition<std::vector<o2::ctp::CTPDigit>>{InputSpec{"digit", "CTP", "DIGITS", 0}, "CTPDigits", logger})();
}

} // namespace ctp
} // namespace o2
