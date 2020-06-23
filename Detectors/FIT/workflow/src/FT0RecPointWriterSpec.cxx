// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0RecPointWriterSpec.cxx

#include <vector>

#include "FITWorkflow/FT0RecPointWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsFT0/RecPoints.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
DataProcessorSpec getFT0RecPointWriterSpec(bool useMC)
{
  using RecPointsType = std::vector<o2::ft0::RecPoints>;
  using ChanDataType = std::vector<o2::ft0::ChannelDataFloat>;
  // Spectators for logging
  auto logger = [](RecPointsType const& recPoints) {
    LOG(INFO) << "FT0RecPointWriter pulled " << recPoints.size() << " RecPoints";
  };
  return MakeRootTreeWriterSpec("ft0-recpoint-writer",
                                "o2reco_ft0.root",
                                "o2sim",
                                BranchDefinition<RecPointsType>{InputSpec{"recPoints", "FT0", "RECPOINTS", 0},
                                                                "FT0Cluster",
                                                                "ft0-recpoint-branch-name",
                                                                1,
                                                                logger},
                                BranchDefinition<ChanDataType>{InputSpec{"recChData", "FT0", "RECCHDATA", 0},
                                                               "FT0RecChData",
                                                               "ft0-rechhdata-branch-name"})();
}

} // namespace ft0
} // namespace o2
