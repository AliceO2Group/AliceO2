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
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace indices
{
DECLARE_SOA_INDEX_COLUMN(BC, bc);
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_INDEX_COLUMN(Zdc, zdc);
} // namespace indices

DECLARE_SOA_INDEX_TABLE(MatchedExclusive, BCs, "MATCHED", indices::CollisionId, indices::BCId, indices::ZdcId);
DECLARE_SOA_INDEX_TABLE(MatchedSparse, BCs, "MATCHED", indices::CollisionId, indices::BCId, indices::ZdcId);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  Builds<aod::MatchedExclusive> matched_e;
  Builds<aod::MatchedSparse> matched_s;
  void init(o2::framework::InitContext&)
  {
  }
};

struct BTask {
  void process(aod::MatchedExclusive::iterator const& m, aod::Zdcs const&, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Collision %d; Ntrk: %d", m.collisionId(), tracks.size());
    LOGF(INFO, "ZDC: E1 = %.3f; E2 = %.3f", m.zdc().energyZEM1(), m.zdc().energyZEM2());
    auto t1 = tracks.begin();
    auto t2 = t1 + (tracks.size() - 1);
    LOGF(INFO, "track 1 from %d; track %d from %d", t1.collisionId(), tracks.size(), t2.collisionId());
  }
};

struct CTask {
  void process(aod::MatchedSparse::iterator const& m, aod::Zdcs const&, aod::Tracks const& tracks)
  {
    LOGF(INFO, "Collision %d; Ntrk: %d", m.collisionId(), tracks.size());
    if (m.has_zdc()) {
      LOGF(INFO, "ZDC: E1 = %.3f; E2 = %.3f", m.zdc().energyZEM1(), m.zdc().energyZEM2());
    } else {
      LOGF(INFO, "No ZDC info");
    }
    auto t1 = tracks.begin();
    auto t2 = t1 + (tracks.size() - 1);
    LOGF(INFO, "track 1 from %d; track %d from %d", t1.collisionId(), tracks.size(), t2.collisionId());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-index"),
    adaptAnalysisTask<BTask>("consume-index-exclusive"),
    adaptAnalysisTask<CTask>("consume-index-sparse")};
}
