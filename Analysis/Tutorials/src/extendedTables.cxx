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
/// \brief Extending existing tables with expression and dynamic columns.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

namespace o2::aod
{
namespace extension
{
DECLARE_SOA_EXPRESSION_COLUMN(P2exp, p2exp, float, track::p* track::p);

DECLARE_SOA_COLUMN(mX, mx, float);
DECLARE_SOA_COLUMN(mY, my, float);
DECLARE_SOA_COLUMN(mP, mp, float);
DECLARE_SOA_DYNAMIC_COLUMN(P2dyn, p2dyn, [](float p) -> float { return p * p; });
DECLARE_SOA_DYNAMIC_COLUMN(R2dyn, r2dyn, [](float x, float y) -> float { return x * x + y * y; });
} // namespace extension

DECLARE_SOA_TABLE(DynTable, "AOD", "DYNTABLE",
                  extension::mX, extension::mY, extension::mP,
                  extension::P2dyn<extension::mP>,
                  extension::R2dyn<extension::mX, extension::mY>);

DECLARE_SOA_EXTENDED_TABLE_USER(ExTable, Tracks, "EXTABLE",
                                extension::P2exp);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// Extend table Tracks with expression column
struct ExtendTable {
  // group tracks according to collisions
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    // add expression column o2::aod::extension::P2exp to table
    // o2::aod::Tracks
    auto table_extension = soa::Extend<aod::Tracks, aod::extension::P2exp>(tracks);

    // loop over the rows of the new table
    for (auto& row : table_extension) {
      if (row.trackType() != 3) {
        if (row.index() % 10000 == 0) {
          LOGF(info, "EXPRESSION Pt^2 = %.3f", row.p2exp());
        }
      }
    }
  }
};

// Attach dynamic columns to table Tracks
struct AttachColumn {
  // group tracks according to collisions
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    // add dynamic columns o2::aod::extension::P2dyn and
    // o2::aod::extension::R2dyn to table o2::aod::Tracks
    auto table_extension = soa::Attach<aod::Tracks,
                                       aod::extension::R2dyn<aod::track::X, aod::track::Y>>(tracks);
    // loop over the rows of the new table
    for (auto& row : table_extension) {
      if (row.trackType() != 3) {
        if (row.index() % 10000 == 0) {
          LOGF(info, "DYNAMIC R^2 = %.3f", row.r2dyn());
        }
      }
    }
  }
};

// extend and attach within process function
struct ExtendAndAttach {
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {

    // combine Extend and Attach to create a new table
    auto table_extension = soa::Extend<aod::Tracks, aod::extension::P2exp>(tracks);
    auto table_attached = soa::Attach<decltype(table_extension),
                                      aod::extension::P2dyn<aod::track::P>,
                                      aod::extension::R2dyn<aod::track::X, aod::track::Y>>(table_extension);

    // loop over the rows of the new table
    for (auto& row : table_attached) {
      if (row.trackType() != 3) {
        if (row.index() % 10000 == 0) {
          LOGF(info, "C: EXPRESSION P^2 = %.3f, DYNAMIC P^2 = %.3f R^2 = %.3f", row.p2exp(), row.p2dyn(), row.r2dyn());
        }
      }
    }
  }
};

// spawn ExTable and produce DynTable
struct SpawnDynamicColumns {
  Produces<aod::DynTable> dyntable;
  Spawns<aod::ExTable> extable;

  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      dyntable(track.x(), track.y(), track.p());
    }
  }
};

// loop over the joined table <ExTable, DynTable>
struct ProcessExtendedTables {
  // join the table ExTable and DynTable
  using allinfo = soa::Join<aod::ExTable, aod::DynTable>;

  void process(aod::Collision const&, allinfo const& tracks)
  {
    // loop over the rows of the new table
    for (auto& row : tracks) {
      if (row.trackType() != 3) {
        if (row.index() % 10000 == 0) {
          LOGF(info, "E: EXPRESSION P^2 = %.3f, DYNAMIC P^2 = %.3f R^2 = %.3f", row.p2exp(), row.p2dyn(), row.r2dyn());
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ExtendTable>(cfgc),
    adaptAnalysisTask<AttachColumn>(cfgc),
    adaptAnalysisTask<ExtendAndAttach>(cfgc),
    adaptAnalysisTask<SpawnDynamicColumns>(cfgc),
    adaptAnalysisTask<ProcessExtendedTables>(cfgc),
  };
}
