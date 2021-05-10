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
///
/// \brief Write a table to a root tree.
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"

namespace o2::aod
{
namespace minmax
{
DECLARE_SOA_COLUMN(Minpt, minpt, float);
DECLARE_SOA_COLUMN(Maxpt, maxpt, float);
DECLARE_SOA_COLUMN(Mineta, mineta, float);
DECLARE_SOA_COLUMN(Maxeta, maxeta, float);
} // namespace minmax

DECLARE_SOA_TABLE(MinMaxPt, "AOD", "MINMAXPT",
                  minmax::Minpt, minmax::Maxpt);

DECLARE_SOA_TABLE(MinMaxEta, "AOD", "MINMAXETA",
                  minmax::Mineta, minmax::Maxeta);

} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct ATask {
  Produces<aod::MinMaxPt> minmaxpt;
  float minpt;
  float maxpt;

  void process(aod::Collision const& collision, aod::Tracks const& tracks)
  {
    // reset minpt / maxpt
    minpt = 1.E3;
    maxpt = 0.;

    // loop over tracks of the collision
    LOGF(info, "Collision %d number of tracks %d", collision.index(), tracks.size());
    for (auto& track : tracks) {
      if (track.pt() < minpt) {
        minpt = track.pt();
      }
      if (track.pt() > maxpt) {
        maxpt = track.pt();
      }
    }
    LOGF(info, "  ptmin %f ptmax %f", minpt, maxpt);

    // update table minmax
    minmaxpt(minpt, maxpt);
  }
};

struct BTask {
  Produces<aod::MinMaxEta> minmaxeta;
  float mineta;
  float maxeta;

  void process(aod::Collision const& collision, aod::Tracks const& tracks)
  {
    // reset mineta / maxeta
    mineta = 1.E3;
    maxeta = -1.E3;

    // loop over tracks of the collision
    LOGF(info, "Collision %d number of tracks %d", collision.index(), tracks.size());
    for (auto& track : tracks) {
      if (track.eta() < mineta) {
        mineta = track.eta();
      }
      if (track.eta() > maxeta) {
        maxeta = track.eta();
      }
    }

    // update table minmax
    minmaxeta(mineta, maxeta);
  }
};

struct CTask {
  void process(aod::MinMaxEta const& minmaxetas)
  {
    for (auto& minmaxeta : minmaxetas) {
      LOGF(info, "  etamin %f etamax %f", minmaxeta.mineta(), minmaxeta.maxeta());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>(cfgc, TaskName{"aod-writer-tutorial_A"}),
    adaptAnalysisTask<BTask>(cfgc, TaskName{"aod-writer-tutorial_B"}),
    adaptAnalysisTask<CTask>(cfgc, TaskName{"aod-writer-tutorial_C"}),
  };
}
