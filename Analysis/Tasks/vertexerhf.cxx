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
#include "Analysis/SecondaryVertex.h"
#include "DetectorsBase/DCAFitter.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>

struct CandidateBuilding2Prong {
  // secondary vertex position
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hchi2dca{TH1F("hchi2dca", "chi2 DCA decay", 1000, 0., 0.0002)};
  // primary vertex position
  OutputObj<TH1F> hvtxp_x_out{TH1F("hvertexx", "x primary vtx", 100, -10., 10.)};
  OutputObj<TH1F> hvtxp_y_out{TH1F("hvertexy", "y primary vtx", 100, -10., 10.)};
  OutputObj<TH1F> hvtxp_z_out{TH1F("hvertexz", "z primary vtx", 100, -10., 10.)};
  // track distributions before cuts
  OutputObj<TH1F> hpt_nocuts{TH1F("hpt_nocuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> htgl_nocuts{TH1F("htgl_nocuts", "tgl tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hitsmap_nocuts{TH1F("hitsmap_nocuts", "hitsmap", 100, 0., 100.)};
  // track distributions after cuts
  OutputObj<TH1F> hpt_cuts{TH1F("hpt_cuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> htgl_cuts{TH1F("htgl_cuts", "tgl tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hitsmap_cuts{TH1F("hitsmap_cuts", "hitsmap", 100, 0., 100.)};

  Produces<aod::SecVtx2Prong> secvtx2prong;

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
      //fill track distribution before selection
      hitsmap_nocuts->Fill(clustermap_0);
      hpt_nocuts->Fill(track_0.pt());
      htgl_nocuts->Fill(track_0.tgl());
      bool isselected_0 = track_0.tpcNCls() > 70 && track_0.flags() & 0x4 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        continue;
      //fill track distribution after selection
      hitsmap_cuts->Fill(clustermap_0);
      hpt_cuts->Fill(track_0.pt());
      htgl_cuts->Fill(track_0.tgl());

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
        if (track_0.signed1Pt() * track_1.signed1Pt() > 0)
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
          secvtx2prong(track_0.collisionId(), collision.posX(), collision.posY(), collision.posZ(), vtx.x, vtx.y, vtx.z, track_0.globalIndex(),
                       pvec0[0], pvec0[1], pvec0[2], track_1.globalIndex(), pvec1[0], pvec1[1], pvec1[2], ic, mass_, masssw_);
          hchi2dca->Fill(df.getChi2AtPCACandidate(ic));

          //float declengxy = decaylengthXY(secVtx2prong.posX(), secVtx2prong.posY(), secVtx2prong.posdecayx(), secVtx2prong.posdecayy());
          //float declengxyz = decaylength(secVtx2prong.posX(), secVtx2prong.posY(), secVtx2prong.posZ(),
          //                           secVtx2prong.posdecayx(), secVtx2prong.posdecayy(), secVtx2prong.posdecayz());
        }
      }
    }
  }
};

struct CandidateBuildingDzero {
  Produces<aod::Cand2Prong> cand2prong;
  void process(aod::SecVtx2Prong const& secVtx2Prongs, aod::Tracks const& tracks) // HERE IT WHAT WORKS
  //void process(aod::SecVtx2Prong const& secVtx2Prongs)  //THE SIMPLE LOOP WORKS AS WELL OF COURSE

  //BELOW IS WHAT I WOULD LIKE TO BE ABLE TO DO AND THAT IS STILL NOT WORKING
  //void process(aod::SecVtx2Prong const& secVtx2Prongs, soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    LOGF(info, "NEW EVENT");

    for (auto& secVtx2prong : secVtx2Prongs) {
      //example to access the track at the index saved in the secVtx2Prongs
      LOGF(INFO, " I am now accessing track information looping over secondary vertices  %f", (tracks.begin() + secVtx2prong.index0()).y());
      float masspion = 0.140;
      float masskaon = 0.494;
      float mass_ = invmass2prongs(secVtx2prong.px0(), secVtx2prong.py0(), secVtx2prong.pz0(), masspion,
                                   secVtx2prong.px1(), secVtx2prong.py1(), secVtx2prong.pz1(), masskaon);
      float masssw_ = invmass2prongs(secVtx2prong.px0(), secVtx2prong.py0(), secVtx2prong.pz0(), masskaon,
                                     secVtx2prong.px1(), secVtx2prong.py1(), secVtx2prong.pz1(), masspion);
      cand2prong(secVtx2prong.collisionId(), mass_, masssw_);
    }
  }
};

struct DzeroHistoTask {
  OutputObj<TH1F> hmass_nocuts_out{TH1F("hmass_nocuts", "2-track inv mass", 500, 0, 5.0)};
  OutputObj<TH1F> hdecayxy{TH1F("hdecayxy", "decay length xy", 100, 0., 1.0)};
  OutputObj<TH1F> hdecayxyz{TH1F("hdecayxyz", "decay length", 100, 0., 1.0)};

  void process(aod::Cand2Prong const& cand2Prongs, aod::SecVtx2Prong const& secVtx2Prongs)
  {
    LOGF(info, "NEW EVENT");

    for (auto& secVtx2prong : secVtx2Prongs) {
      hdecayxy->Fill(secVtx2prong.decaylengthXY());
      hdecayxyz->Fill(secVtx2prong.decaylength());
      hmass_nocuts_out->Fill(secVtx2prong.mass());
      hmass_nocuts_out->Fill(secVtx2prong.massbar());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<CandidateBuilding2Prong>("vertexerhf-candidatebuilding2prong"),
    adaptAnalysisTask<CandidateBuildingDzero>("vertexerhf-candidatebuildingDzero"),
    adaptAnalysisTask<DzeroHistoTask>("vertexerhf-Dzerotask")};
}
