// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ZDCRecoWriterDPLSpec.cxx

#include <vector>

#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsZDC/BCRecData.h"

#include "ZDCWorkflow/ZDCRecoWriterDPLSpec.h"
using namespace o2::framework;

namespace o2
{
namespace zdc
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
// Unused arguments: bool mctruth, bool simVersion
DataProcessorSpec getZDCRecoWriterDPLSpec()
{
  std::string writerName = "ZDCRecoWriter";
  std::string fnameDef = "zdcreco.root";

  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  return MakeRootTreeWriterSpec(writerName.data(),
                                fnameDef.data(),
                                "o2rec",
                                BranchDefinition<std::vector<o2::zdc::BCRecData>>{InputSpec{"bcrec", "ZDC", "BCREC"}, "ZDCRecBC"},
                                BranchDefinition<std::vector<o2::zdc::BCRecData>>{InputSpec{"energy", "ZDC", "ENERGY"}, "ZDCRecE"},
                                BranchDefinition<std::vector<o2::zdc::BCRecData>>{InputSpec{"tdcdata", "ZDC", "TDCDATA"}, "ZDCRecTDC"})();
}

} // namespace zdc
} // namespace o2
