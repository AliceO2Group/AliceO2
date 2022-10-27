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

/// \file DigitWriterSpec.cxx
/// \author Roman Lietava

#include "CTPWorkflowIO/DigitWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/LumiInfo.h"

namespace o2
{
namespace ctp
{
template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

framework::DataProcessorSpec getDigitWriterSpec(bool raw)
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  // Spectators for logging
  auto logger = [](std::vector<o2::ctp::CTPDigit> const& vecDigits) {
    LOG(info) << "CTPDigitWriter pulled " << vecDigits.size() << " digits";
  };
  if (raw) {
    return MakeRootTreeWriterSpec("ctp-digit-writer-dec", "ctpdigits.root",
                                  MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CTP digits/Lumi"},
                                  BranchDefinition<std::vector<o2::ctp::CTPDigit>>{InputSpec{"digit", "CTP", "DIGITS", 0}, "CTPDigits", logger},
                                  BranchDefinition<o2::ctp::LumiInfo>{InputSpec{"lumi", "CTP", "LUMI", 0}, "CTPLumi"})();
  }
  // MC digits case, no lumi available
  return MakeRootTreeWriterSpec("ctp-digit-writer", "ctpdigits.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CTP digits"},
                                BranchDefinition<std::vector<o2::ctp::CTPDigit>>{InputSpec{"digit", "CTP", "DIGITS", 0}, "CTPDigits", logger})();
}

} // namespace ctp
} // namespace o2
