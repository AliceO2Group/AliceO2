// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//
/// \brief Extending existing tables with expression and dynamic columns.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

#declare expression column P2
namespace o2::aod
{
namespace extension
{
DECLARE_SOA_EXPRESSION_COLUMN(P2, p2, float, track::p* track::p);
} // namespace extension
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct ATask {
  // group tracks according to collisions
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    // add expression column o2::aod::extension::P2exp to table
    // o2::aod::Tracks
    auto table_extension = soa::Extend<aod::Tracks, aod::extension::P2>(tracks);
    for (auto& row : table_extension) {
      if (row.trackType() != 3) {
        if (row.index() % 100 == 0) {
          LOGF(info, "P^2 = %.3f", row.p2());
        }
      }
    }
  }
};

namespace o2::aod
{
DECLARE_SOA_EXTENDED_TABLE_USER(MTracks, aod::Tracks, "MTRACK", aod::track::extension::P);
} // namespace o2::aod

struct BTask {
  Spawns<aod::MTracks> mtracks;

  void process(aod::Collision const&, aod::MTracks const& mtracks)
  {
    auto table_extension = soa::Extend<aod::Tracks, aod::extension::P2>(tracks);
    for (auto& row : table_extension) {
      if (row.trackType() != 3) {
        if (row.index() % 100 == 0) {
          LOGF(info, "P^2 = %.3f", row.p2());
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  // create and use table
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, TaskName{"extend-columns-tutorial_A"}),
    adaptAnalysisTask<BTask>(cfgc, TaskName{"extend-columns-tutorial_B"}),
  };
}
