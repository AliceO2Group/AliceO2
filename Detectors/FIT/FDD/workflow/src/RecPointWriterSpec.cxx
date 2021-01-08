// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecPointWriterSpec.cxx

#include <vector>

#include "FDDWorkflow/RecPointWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsFDD/RecPoint.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
DataProcessorSpec getFDDRecPointWriterSpec(bool useMC)
{
  using RecPointsType = std::vector<o2::fdd::RecPoint>;
  // Spectators for logging
  auto logger = [](RecPointsType const& recPoints) {
    LOG(INFO) << "FDDRecPointWriter pulled " << recPoints.size() << " RecPoints";
  };
  return MakeRootTreeWriterSpec("fdd-recpoint-writer",
                                "o2reco_fdd.root",
                                "o2sim",
                                BranchDefinition<RecPointsType>{InputSpec{"recPoints", "FDD", "RECPOINTS", 0},
                                                                "FDDCluster",
                                                                "fdd-recpoint-branch-name",
                                                                1,
                                                                logger})();
}

} // namespace fdd
} // namespace o2
