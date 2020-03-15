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
namespace etaphi
{
DECLARE_SOA_COLUMN(Eta, eta2, float, "fEta");
DECLARE_SOA_COLUMN(Phi, phi2, float, "fPhi");
} // namespace etaphi
DECLARE_SOA_TABLE(EtaPhi, "AOD", "ETAPHI",
                  etaphi::Eta, etaphi::Phi);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  Produces<aod::EtaPhi> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      etaphi(track.eta(), track.phi());
    }
  }
};

struct BTask {
  Filter flt = (aod::etaphi::eta2 < 1.0f) && (aod::etaphi::eta2 > -1.0f) && (aod::etaphi::phi2 < 2.0f) && (aod::etaphi::phi2 > 1.0f);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::EtaPhi>> const& tracks)
  {
    LOGF(INFO, "Collision: %d [N = %d]", collision.globalIndex(), tracks.size());
    for (auto& track : tracks) {
      LOGF(INFO, "id = %d; eta:  -1 < %.3f < 1; phi: 1 < %.3f < 2", track.collisionId(), track.eta2(), track.phi2());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
    adaptAnalysisTask<BTask>("consume-etaphi")};
}
