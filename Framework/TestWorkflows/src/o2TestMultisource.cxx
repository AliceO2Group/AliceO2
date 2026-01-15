// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief Tests that the same tables from different origins are routed correctly.
///        Requires two input files, <name>.root and <name>_EMB.root, that contain
///        same number of DFs with the same names.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

namespace o2::aod
{
O2ORIGIN("EMB");
template <is_aod_hash T>
using BCsFrom = BCs_001From<T>;
using TracksPlus = soa::Join<StoredTracksIU, StoredTracksExtra>;
template <is_aod_hash T>
using TracksPlusFrom = soa::Join<StoredTracksIUFrom<T>, StoredTracksExtra_002From<T>>;
} // namespace o2::aod

struct TestEmbeddingSubscription {
  void process(aod::BCs const& bcs, aod::BCsFrom<aod::Hash<"EMB"_h>> const& bcse,
               aod::TracksPlus const& tracks, aod::TracksPlusFrom<aod::Hash<"EMB"_h>> const& trackse)
  {
    LOGP(info, "BCs from run {} and {}", bcs.begin().runNumber(), bcse.begin().runNumber());
    LOGP(info, "Joined tracks: {} and {}", tracks.size(), trackse.size());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return {adaptAnalysisTask<TestEmbeddingSubscription>(cfgc)};
}
