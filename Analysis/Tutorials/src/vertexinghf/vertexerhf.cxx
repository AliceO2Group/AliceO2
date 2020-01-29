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
#include "DetectorsBase/DCAFitter.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
namespace o2::aod
{
namespace etaphi
{
DECLARE_SOA_COLUMN(Eta, etas, float, "fEta");
DECLARE_SOA_COLUMN(Phi, phis, float, "fPhi");
} // namespace etaphi
namespace secvtx
{
DECLARE_SOA_COLUMN(Posx, posx, float, "fPosx");
DECLARE_SOA_COLUMN(Posy, posy, float, "fPosy");
DECLARE_SOA_COLUMN(Index0, index0, int, "fIndex0");
DECLARE_SOA_COLUMN(Index1, index1, int, "fIndex1");
DECLARE_SOA_COLUMN(Index2, index2, int, "fIndex2");
DECLARE_SOA_COLUMN(Tracky0, tracky0, float, "fTracky0");
DECLARE_SOA_COLUMN(Tracky1, tracky1, float, "fTracky1");
DECLARE_SOA_COLUMN(Tracky2, tracky2, float, "fTracky2");

} // namespace secvtx
namespace cand2prong
{
DECLARE_SOA_COLUMN(Mass, mass, float, "fMass");
} // namespace cand2prong

DECLARE_SOA_TABLE(EtaPhi, "RN2", "ETAPHI",
                  etaphi::Eta, etaphi::Phi);
DECLARE_SOA_TABLE(SecVtx, "AOD", "SECVTX",
                  secvtx::Posx, secvtx::Posy, secvtx::Index0, secvtx::Index1, secvtx::Index2, secvtx::Tracky0, secvtx::Tracky1, secvtx::Tracky2);
DECLARE_SOA_TABLE(Cand2Prong, "AOD", "CAND2PRONG",
                  cand2prong::Mass);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct ATask {
  Produces<aod::EtaPhi> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      auto phi = asin(track.snp()) + track.alpha() + M_PI;
      auto eta = log(tan(0.25 * M_PI - 0.5 * atan(track.tgl())));

      etaphi(phi, eta);
    }
  }
};
struct VertexerHFTask {
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 100, -0.1, 0.1)};
  Produces<aod::SecVtx> secvtx;

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksCov> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d", tracks.size());
    o2::base::DCAFitter df(5.0, 10.);

    for (auto it_0 = tracks.begin(); it_0 != tracks.end(); ++it_0) {
      auto& track_0 = *it_0;
      if (track_0.pt() < 10.)
        continue;
      float x0_ = track_0.x();
      float alpha0_ = track_0.alpha();
      std::array<float, 5> arraypar0 = {track_0.y(), track_0.z(), track_0.snp(), track_0.tgl(), track_0.signed1Pt()};
      std::array<float, 15> covpar0 = {track_0.cYY(), track_0.cZY(), track_0.cZZ(), track_0.cSnpY(), track_0.cSnpZ(),
                                       track_0.cSnpSnp(), track_0.cTglY(), track_0.cTglZ(), track_0.cTglSnp(), track_0.cTglTgl(),
                                       track_0.c1PtY(), track_0.c1PtZ(), track_0.c1PtSnp(), track_0.c1PtTgl(), track_0.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar0(x0_, alpha0_, arraypar0, covpar0);

      for (auto it_1 = it_0 + 1; it_1 != tracks.end(); ++it_1) {
        auto& track_1 = *it_1;
        if (track_1.pt() < 10.)
          continue;
        float x1_ = track_1.x();
        float alpha1_ = track_1.alpha();
        std::array<float, 5> arraypar1 = {track_1.y(), track_1.z(), track_1.snp(), track_1.tgl(), track_1.signed1Pt()};
        std::array<float, 15> covpar1 = {track_1.cYY(), track_1.cZY(), track_1.cZZ(), track_1.cSnpY(), track_1.cSnpZ(),
                                         track_1.cSnpSnp(), track_1.cTglY(), track_1.cTglZ(), track_1.cTglSnp(), track_1.cTglTgl(),
                                         track_1.c1PtY(), track_1.c1PtZ(), track_1.c1PtSnp(), track_1.c1PtTgl(), track_1.c1Pt21Pt2()};
        o2::track::TrackParCov trackparvar1(x1_, alpha1_, arraypar1, covpar1);

        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar0, trackparvar1);
        //        printf("\n\nTesting with abs DCA minimization: %d candidates found\n", nCand);
        for (int ic = 0; ic < nCand; ic++) {
          const o2::base::DCAFitter::Triplet& vtx = df.getPCACandidate(ic);
          //	  LOGF(info, "print %f", vtx.x);
          hvtx_x_out->Fill(vtx.x);
          hvtx_y_out->Fill(vtx.y);
          hvtx_z_out->Fill(vtx.z);
          secvtx(vtx.x, vtx.y, track_0.index(), track_1.index(), -1., track_0.y(), track_1.y(), -1.);
          //	  LOGF(info, "Track labels: %d, %d", track_0.index(), track_1.index());
        }
      }
    }
  }
};

struct CandidateBuilder2Prong {
  Produces<aod::Cand2Prong> cand2prong;
  void process(aod::SecVtx const& secVtxs, aod::Tracks const& tracks)
  {
    LOGF(info, "NEW EVENT");
    for (auto& secVtx : secVtxs) {
      LOGF(INFO, "Consume the table (%f, %f, %f, %f)", secVtx.posx(), secVtx.posy(), secVtx.tracky0(), (tracks.begin() + secVtx.index0()).y());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
    adaptAnalysisTask<VertexerHFTask>("vertexerhf-task"),
    adaptAnalysisTask<CandidateBuilder2Prong>("skimvtxtable-task")};
}
