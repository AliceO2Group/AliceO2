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

/// @file   RecPointWriterSpec.cxx

#include <vector>

#include "FV0Workflow/RecPointWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsFV0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
DataProcessorSpec getRecPointWriterSpec(bool useMC)
{
  using RecPointsType = std::vector<o2::fv0::RecPoints>;
  using ChanDataType = std::vector<o2::fv0::ChannelDataFloat>;
  // Spectators for logging
  auto logger = [](RecPointsType const& recPoints) {
    LOG(INFO) << "FV0RecPointWriter pulled " << recPoints.size() << " RecPoints";
  };
  return MakeRootTreeWriterSpec("fv0-recpoint-writer",
                                "o2reco_fv0.root",
                                "o2sim",
                                BranchDefinition<RecPointsType>{InputSpec{"recPoints", "FV0", "RECPOINTS", 0},
                                                                "FV0Cluster",
                                                                "fv0-recpoint-branch-name",
                                                                1,
                                                                logger},
                                BranchDefinition<ChanDataType>{InputSpec{"recChData", "FV0", "RECCHDATA", 0},
                                                               "FV0RecChData",
                                                               "fv0-rechhdata-branch-name"})();
}

} // namespace fv0
} // namespace o2
