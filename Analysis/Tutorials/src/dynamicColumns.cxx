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
namespace extension
{
DECLARE_SOA_DYNAMIC_COLUMN(P2, p2, [](float p) { return p * p; });
} // namespace etaphi
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct ATask {
  void process(aod::Collision const&, aod::Tracks const& tracks)
  {
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

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  // create and use table
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("attach-showcase")};
}
