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
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/PedestalData.h"

#include "ZDCWorkflow/ZDCDigitWriterDPLSpec.h"
using namespace o2::framework;

namespace o2
{
namespace zdc
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
DataProcessorSpec getZDCDigitWriterDPLSpec()
{
//   // Spectators for logging
//   auto logger = [](DigitType const& digits) {
//     LOG(INFO) << "FT0DigitWriter pulled " << digits.size() << " digits";
//   };
  return MakeRootTreeWriterSpec(
    "zdc-digit-writer", "o2digit_zdc.root", "o2sim",
    BranchDefinition<std::vector<o2::zdc::BCData>>{InputSpec{"digitBCinput", "ZDC", "DIGITSBC"}, "ZDCDigitBC"},
                                BranchDefinition<std::vector<o2::zdc::ChannelData>>{InputSpec{"digitChinput", "ZDC", "DIGITSCH"}, "ZDCDigitCh"},
                                BranchDefinition<std::vector<o2::zdc::PedestalData>>{InputSpec{"digitPDinput", "ZDC", "DIGITSPD"}, "ZDCDigitPed"})();
}

} // namespace ft0
} // namespace o2
