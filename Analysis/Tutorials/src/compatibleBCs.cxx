// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief In Run 3 the association between collisions and bunch crossings
///        is not unique as the time of a collision vertex is derived from
///        the track information themselves.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "CommonConstants/LHCConstants.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

template <typename T>
T getCompatibleBCs(aod::Collision const& collision, T const& bcs)
{
  auto bcIter = collision.bc_as<T>();

  // due to the filling scheme the most probably BC may not be the one estimated from the collision time
  uint64_t mostProbableBC = bcIter.globalBC();
  uint64_t meanBC = mostProbableBC - std::lround(collision.collisionTime() / (o2::constants::lhc::LHCBunchSpacingNS / 1000));
  int deltaBC = std::ceil(collision.collisionTimeRes() / (o2::constants::lhc::LHCBunchSpacingNS / 1000) * 4);

  LOGF(INFO, "BC range: %llu - %llu", meanBC - deltaBC, meanBC + deltaBC);

  // find slice of BCs table with BC in [meanBC - deltaBC, meanBC + deltaBC]
  int64_t maxBCId = bcIter.globalIndex();
  int moveCount = 0; // optimize to avoid to re-create the iterator
  while (bcIter != bcs.end() && bcIter.globalBC() <= meanBC + deltaBC && bcIter.globalBC() >= meanBC - deltaBC) {
    LOGF(DEBUG, "Table id %d BC %llu", bcIter.globalIndex(), bcIter.globalBC());
    maxBCId = bcIter.globalIndex();
    ++bcIter;
    ++moveCount;
  }

  bcIter.moveByIndex(-moveCount); // Move back to original position
  int64_t minBCId = collision.bcId();
  while (bcIter != bcs.begin() && bcIter.globalBC() <= meanBC + deltaBC && bcIter.globalBC() >= meanBC - deltaBC) {
    LOGF(DEBUG, "Table id %d BC %llu", bcIter.globalIndex(), bcIter.globalBC());
    minBCId = bcIter.globalIndex();
    --bcIter;
  }

  LOGF(INFO, "Will consider BC entries from %d to %d", minBCId, maxBCId);

  T slice{{bcs.asArrowTable()->Slice(minBCId, maxBCId - minBCId + 1)}, (uint64_t)minBCId};
  bcs.copyIndexBindings(slice);
  return slice;
}

// Example 1 (academic, because it is not enough to just access the BC table):
// This task shows how to loop over the bunch crossings which are compatible with the estimated collision time.
struct CompatibleBCs {
  void process(aod::Collision const& collision, aod::BCs const& bcs)
  {
    LOGF(INFO, "Vertex with most probably BC %llu and collision time %f +- %f ps", collision.bc().globalBC(), collision.collisionTime(), collision.collisionTimeRes());

    auto bcSlice = getCompatibleBCs(collision, bcs);

    for (auto& bc : bcSlice) {
      LOGF(info, "This collision may belong to BC %lld", bc.globalBC());
    }
  }
};

// Example 2:
// Using getCompatibleBCs to retrieve the entries of several tables linked through the BC table (here FT0 and FV0A)
// and making sure that one has the direct association between them
// Note that one has to subscribe to aod::FT0s and aod::FV0As to load
// the relevant data even if you access the data itself through m.ft0() and m.fv0a()
struct CompatibleT0V0A {
  void process(aod::Collision const& collision, soa::Join<aod::BCs, aod::Run3MatchedToBCSparse> const& bct0s, aod::FT0s& ft0s, aod::FV0As& fv0as)
  {
    // NOTE collision.bc() causes SEGV here because we have only subscribed to BCs joined, therefore:
    auto bc = collision.bc_as<soa::Join<aod::BCs, aod::Run3MatchedToBCSparse>>();
    LOGF(INFO, "Vertex with most probable BC %llu and collision time %f +- %f ps", bc.globalBC(), collision.collisionTime(), collision.collisionTimeRes());

    auto bcSlice = getCompatibleBCs(collision, bct0s);

    for (auto& bc : bcSlice) {
      if (bc.has_ft0() && bc.has_fv0a()) {
        LOGF(info, "This collision may belong to BC %lld and has T0 timeA: %f and V0A time: %f", bc.globalBC(), bc.ft0().timeA(), bc.fv0a().time());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<CompatibleBCs>(cfgc),
    adaptAnalysisTask<CompatibleT0V0A>(cfgc),
  };
}
