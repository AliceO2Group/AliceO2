// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/ASoAHelpers.h"

namespace o2::aod
{
namespace hash
{
DECLARE_SOA_COLUMN(Bin, bin, int);
} // namespace hash
DECLARE_SOA_TABLE(Hashes, "AOD", "HASH", hash::Bin);

using Hash = Hashes::iterator;
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::soa;

// This is a tentative workflow to get mixed-event tracks
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that

struct HashTask {
  std::vector<float> xBins{-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  std::vector<float> yBins{-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
  Produces<aod::Hashes> hashes;

  // Calculate hash for an element based on 2 properties and their bins.
  int getHash(const std::vector<float>& xBins, const std::vector<float>& yBins, float colX, float colY)
  {
    for (int i = 0; i < xBins.size(); i++) {
      if (colX < xBins[i]) {
        for (int j = 0; j < yBins.size(); j++) {
          if (colY < yBins[j]) {
            return i + j * (xBins.size() + 1);
          }
        }
        // overflow for yBins only
        return i + yBins.size() * (xBins.size() + 1);
      }
    }

    // overflow for xBins only
    for (int j = 0; j < yBins.size(); j++) {
      if (colY < yBins[j]) {
        return xBins.size() + j * (xBins.size() + 1);
      }
    }

    // overflow for both bins
    return (yBins.size() + 1) * (xBins.size() + 1) - 1;
  }

  void process(aod::Collisions const& collisions)
  {
    for (auto& collision : collisions) {
      int hash = getHash(xBins, yBins, collision.posX(), collision.posY());
      LOGF(info, "Collision: %d (%f, %f, %f) hash: %d", collision.index(), collision.posX(), collision.posY(), collision.posZ(), hash);
      hashes(hash);
    }
  }
};

struct CollisionsCombinationsTask {
  void process(const aod::Hashes& hashes, aod::Collisions& collisions, aod::Tracks& tracks)
  {
    collisions.bindExternalIndices(&tracks);
    auto tracksTuple = std::make_tuple(tracks);
    AnalysisDataProcessorBuilder::GroupSlicer slicer(collisions, tracksTuple);

    // Strictly upper categorised collisions
    for (auto& [c1, c2] : selfCombinations("fBin", join(hashes, collisions), join(hashes, collisions))) {
      LOGF(info, "Collisions bin: %d pair: %d (%f, %f, %f), %d (%f, %f, %f)", c1.bin(), c1.index(), c1.posX(), c1.posY(), c1.posZ(), c2.index(), c2.posX(), c2.posY(), c2.posZ());

      auto it1 = slicer.begin();
      auto it2 = slicer.begin();
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == c1.index()) {
          it1 = slice;
          break;
        }
      }
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == c2.index()) {
          it2 = slice;
          break;
        }
      }
      auto tracks1 = std::get<aod::Tracks>(it1.associatedTables());
      tracks1.bindExternalIndices(&collisions);
      auto tracks2 = std::get<aod::Tracks>(it2.associatedTables());
      tracks2.bindExternalIndices(&collisions);

      for (auto& [t1, t2] : combinations(CombinationsFullIndexPolicy(tracks1, tracks2))) {
        LOGF(info, "Mixed event tracks pair: (%d, %d) from events (%d, %d)", t1.index(), t2.index(), c1.index(), c2.index());
      }
    }
  }
};

// What we would like to have
struct MixedEventsTask {
  void process(aod::Collision const& col1, aod::Tracks const& tracks1, aod::Collision const& col2, aod::Tracks const& tracks2)
  {
    for (auto& [t1, t2] : combinations(CombinationsFullIndexPolicy(tracks1, tracks2))) {
      LOGF(info, "Mixed event tracks pair: (%d, %d) from events (%d, %d)", t1.index(), t2.index(), c1.index(), c2.index());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HashTask>("collisions-hashed"),
    adaptAnalysisTask<CollisionsCombinationsTask>("mixed-event-tracks")};
}
