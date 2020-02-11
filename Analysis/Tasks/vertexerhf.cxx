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
namespace secvtx
{
DECLARE_SOA_COLUMN(dPosx, dposx, float, "fdPosx");
DECLARE_SOA_COLUMN(dPosy, dposy, float, "fdPosy");
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

DECLARE_SOA_TABLE(SecVtx, "AOD", "SECVTX",
                  secvtx::dPosx, secvtx::dPosy, secvtx::Index0, secvtx::Index1, secvtx::Index2, secvtx::Tracky0, secvtx::Tracky1, secvtx::Tracky2);
DECLARE_SOA_TABLE(Cand2Prong, "AOD", "CAND2PRONG",
                  cand2prong::Mass);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

float decaylengthXY(float xvtxp, float yvtxp, float xvtxd, float yvtxd)
{
  float decl_ = sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp));
  return decl_;
};

float decaylength(float xvtxp, float yvtxp, float zvtxp, float xvtxd, float yvtxd, float zvtxd)
{
  float decl_ = sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp) + (zvtxd - zvtxp) * (zvtxd - zvtxp));
  return decl_;
};

float energy(float px, float py, float pz, float mass)
{
  float en_ = sqrtf(mass * mass + px * px + py * py + pz * pz);
  return en_;
};

float invmass2prongs(float px0, float py0, float pz0, float mass0,
                     float px1, float py1, float pz1, float mass1)
{

  float energy0_ = energy(px0, py0, pz0, mass0);
  float energy1_ = energy(px1, py1, pz1, mass1);
  float energytot = energy0_ + energy1_;

  float psum2 = (px0 + px1) * (px0 + px1) + (py0 + py1) * (py0 + py1) + (pz0 + pz1) * (pz0 + pz1);
  float mass = sqrtf(energytot * energytot - psum2);
  return mass;
};

struct VertexerHFTask {
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtxp_x_out{TH1F("hvertexx", "x primary vtx", 100, -10., 10.)};
  OutputObj<TH1F> hvtxp_y_out{TH1F("hvertexy", "y primary vtx", 100, -10., 10.)};
  OutputObj<TH1F> hvtxp_z_out{TH1F("hvertexz", "z primary vtx", 100, -10., 10.)};
  OutputObj<TH1F> hmass_out{TH1F("hmass", "2-track inv mass", 500, 0, 5.0)};
  OutputObj<TH1F> hindex_0_coll{TH1F("hindex_0_coll", "track 0 index coll", 1000000, -0.5, 999999.5)};
  OutputObj<TH1F> hpt_cuts{TH1F("hpt_cuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> htgl_cuts{TH1F("htgl_cuts", "tgl tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hitsmap{TH1F("hitsmap", "hitsmap", 100, 0., 100.)};
  OutputObj<TH1F> heta_cuts{TH1F("heta_cuts", "eta tracks (#GeV)", 100, -2., 2.)};
  OutputObj<TH1F> hdecayxy{TH1F("hdecayxy", "decay length xy", 100, 0., 1.0)};
  OutputObj<TH1F> hdecayxyz{TH1F("hdecayxyz", "decay length", 100, 0., 1.0)};
  OutputObj<TH1F> hchi2dca{TH1F("hchi2dca", "chi2 DCA decay", 1000, 0., 0.0002)};

  Produces<aod::SecVtx> secvtx;

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d", tracks.size());
    o2::base::DCAFitter df(5.0, 10.);
    hvtxp_x_out->Fill(collision.posX());
    hvtxp_y_out->Fill(collision.posY());
    hvtxp_z_out->Fill(collision.posZ());
    for (auto it_0 = tracks.begin(); it_0 != tracks.end(); ++it_0) {
      auto& track_0 = *it_0;
      UChar_t clustermap_0 = track_0.itsClusterMap();
      hitsmap->Fill(clustermap_0);
      bool isselected_0 = track_0.tpcNCls() > 70 && track_0.flags() & 0x4 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        continue;
      hpt_cuts->Fill(track_0.pt());
      htgl_cuts->Fill(track_0.tgl());
      LOGF(info, "globalindex %llu", track_0.globalIndex());
      LOGF(info, "tofChi2 %f", track_0.tofChi2());
      hindex_0_coll->Fill(track_0.globalIndex());
      heta_cuts->Fill(track_0.eta());
      float x0_ = track_0.x();
      float alpha0_ = track_0.alpha();
      std::array<float, 5> arraypar0 = {track_0.y(), track_0.z(), track_0.snp(), track_0.tgl(), track_0.signed1Pt()};
      std::array<float, 15> covpar0 = {track_0.cYY(), track_0.cZY(), track_0.cZZ(), track_0.cSnpY(), track_0.cSnpZ(),
                                       track_0.cSnpSnp(), track_0.cTglY(), track_0.cTglZ(), track_0.cTglSnp(), track_0.cTglTgl(),
                                       track_0.c1PtY(), track_0.c1PtZ(), track_0.c1PtSnp(), track_0.c1PtTgl(), track_0.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar0(x0_, alpha0_, arraypar0, covpar0);

      for (auto it_1 = it_0 + 1; it_1 != tracks.end(); ++it_1) {
        auto& track_1 = *it_1;
        UChar_t clustermap_1 = track_1.itsClusterMap();
        bool isselected_1 = track_1.tpcNCls() > 70 && track_1.flags() & 0x4 && (TESTBIT(clustermap_1, 0) || TESTBIT(clustermap_1, 1));
        if (!isselected_1)
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
        for (int ic = 0; ic < nCand; ic++) {
          const o2::base::DCAFitter::Triplet& vtx = df.getPCACandidate(ic);
          LOGF(info, "vertex x %f", vtx.x);
          hvtx_x_out->Fill(vtx.x);
          hvtx_y_out->Fill(vtx.y);
          hvtx_z_out->Fill(vtx.z);
          o2::track::TrackParCov trackdec0 = df.getTrack0(ic);
          o2::track::TrackParCov trackdec1 = df.getTrack1(ic);
          std::array<float, 3> pvec0;
          std::array<float, 3> pvec1;
          trackdec0.getPxPyPzGlo(pvec0);
          trackdec1.getPxPyPzGlo(pvec1);
          float masspion = 0.140;
          float masskaon = 0.494;
          float mass_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masspion, pvec1[0], pvec1[1], pvec1[2], masskaon);
          float masssw_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masskaon, pvec1[0], pvec1[1], pvec1[2], masspion);
          LOGF(info, "mass_ = %f", mass_);
          LOGF(info, "masssw_ = %f", masssw_);
          secvtx(vtx.x, vtx.y, track_0.globalIndex(), track_1.globalIndex(), -1., track_0.y(), track_1.y(), -1.);
          hmass_out->Fill(mass_);
          hmass_out->Fill(masssw_);
          float declengxy = decaylengthXY(collision.posX(), collision.posY(), vtx.x, vtx.y);
          float declengxyz = decaylength(collision.posX(), collision.posY(), collision.posZ(), vtx.x, vtx.y, vtx.z);
          hdecayxy->Fill(declengxy);
          hdecayxyz->Fill(declengxyz);
          hchi2dca->Fill(df.getChi2AtPCACandidate(ic));
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
      LOGF(INFO, "Consume the table (%f, %f, %f, %f) with track 0 global index %llu", secVtx.dposx(), secVtx.dposy(), secVtx.tracky0(), (tracks.begin() + secVtx.index0()).y(), secVtx.index0());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<VertexerHFTask>("vertexerhf-task")};
  //adaptAnalysisTask<CandidateBuilder2Prong>("skimvtxtable-task")};
}
