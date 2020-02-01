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
DECLARE_SOA_COLUMN(Eta, etas, float, "fEta");
DECLARE_SOA_COLUMN(Phi, phis, float, "fPhi");
DECLARE_SOA_COLUMN(Pt, pts, float, "fPt");
} // namespace etaphi
DECLARE_SOA_TABLE(EtaPhi, "AOD", "ETAPHI",
                  etaphi::Eta, etaphi::Phi);
DECLARE_SOA_TABLE(EtaPhiPt, "AOD", "ETAPHIPT",
                  etaphi::Eta, etaphi::Phi, etaphi::Pt);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

// This is a very simple example showing how to iterate over tracks
// and create a new collection for them.
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that
struct ATask {
  Produces<aod::EtaPhi> etaphi;
  Produces<aod::EtaPhiPt> etaphipt;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));
      float pt = 1.f / track.signed1Pt();
      etaphi(eta, phi);
      etaphipt(eta, phi, pt);
    }
  }
};

struct BTask {
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::EtaPhi> const& extTracks)
  {
    //    LOG(INFO) << "TF: " << collision.timeframeID() << std::endl;
    LOG(INFO) << "Rows: " << extTracks.asArrowTable()->num_rows() << std::endl;
    for (auto& track : extTracks) {
      LOGF(INFO, "(%f, %f) - (%f, %f)", track.eta(), track.phi(), track.etas(), track.phis());
    }
  }
};

struct CTask {
  void process(aod::Collision const& collision, soa::Concat<aod::EtaPhi, aod::EtaPhiPt> const& concatenated)
  {
    //    LOG(INFO) << "TF: " << collision.timeframeID() << std::endl;
    LOG(INFO) << "Rows: " << concatenated.asArrowTable()->num_rows() << std::endl;
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
    adaptAnalysisTask<BTask>("consume-etaphi"),
    adaptAnalysisTask<CTask>("consume-etaphi-twice")};
}
