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
/// \brief Dynamic columns are computed on-the-fly when attached to an existing table
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

namespace o2::aod
{
namespace extension
{
DECLARE_SOA_DYNAMIC_COLUMN(P2, p2, [](float p) { return p * p; });
} // namespace extension
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct AttachColumn {
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
    // dynamic columns are attached to existing tables
    auto table_with_extra_dynamic_columns = soa::Attach<aod::Tracks, aod::extension::P2<aod::track::P>>(tracks);
    for (auto& row : table_with_extra_dynamic_columns) {
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
  return WorkflowSpec{
    adaptAnalysisTask<AttachColumn>(cfgc),
  };
}
