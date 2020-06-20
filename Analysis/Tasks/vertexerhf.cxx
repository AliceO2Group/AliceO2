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
#include "Framework/ASoAHelpers.h"
#include "Analysis/SecondaryVertex.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>

using namespace o2;
using namespace o2::framework;

struct DecayVertexBuilder2Prong {
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
  // secondary vertex position
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hchi2dca{TH1F("hchi2dca", "chi2 DCA decay", 1000, 0., 0.0002)};

  Produces<aod::SecVtx2Prong> secvtx2prong;
  //Configurable<std::string> triggersel{"triggersel", "test", "A string configurable"};
  Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    //LOGP(error, "Trigger selection {}", std::string{triggersel});
    int trigindex = int{triggerindex};
    if (trigindex != -1) {
      LOGF(info, "Selecting on trigger bit %d", trigindex);
      uint64_t triggerMask = collision.bc().triggerMask();
      bool isTriggerClassFired = triggerMask & 1ul << (trigindex - 1);
      if (!isTriggerClassFired)
        return;
    }

    LOGF(info, "N. of Tracks for collision: %d", tracks.size());
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(5.0);
    // After finding the vertex, propagate tracks to the DCA. This is default anyway
    df.setPropagateToPCA(true);
    // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    df.setMaxR(200);
    // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    df.setMaxDZIni(4);
    // stop iterations if max correction is below this value. This is default anyway
    df.setMinParamChange(1e-3);
    // stop iterations if chi2 improves by less that this factor
    df.setMinRelChi2Change(0.9);

    hvtxp_x_out->Fill(collision.posX());
    hvtxp_y_out->Fill(collision.posY());
    hvtxp_z_out->Fill(collision.posZ());
    for (auto it0 = tracks.begin(); it0 != tracks.end(); ++it0) {
      auto& track_0 = *it0;

      UChar_t clustermap_0 = track_0.itsClusterMap();
      //fill track distribution before selection
      hitsmap_nocuts->Fill(clustermap_0);
      hpt_nocuts->Fill(track_0.pt());
      htgl_nocuts->Fill(track_0.tgl());
      bool isselected_0 = track_0.tpcNClsFound() > 70 && track_0.flags() & 0x4;
      isselected_0 = isselected_0 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        continue;
      //fill track distribution after selection
      hitsmap_cuts->Fill(clustermap_0);
      hpt_cuts->Fill(track_0.pt());
      htgl_cuts->Fill(track_0.tgl());

      float x0_ = track_0.x();
      float alpha0_ = track_0.alpha();
      std::array<float, 5> arraypar0 = {track_0.y(), track_0.z(), track_0.snp(),
                                        track_0.tgl(), track_0.signed1Pt()};
      std::array<float, 15> covpar0 = {track_0.cYY(), track_0.cZY(), track_0.cZZ(),
                                       track_0.cSnpY(), track_0.cSnpZ(),
                                       track_0.cSnpSnp(), track_0.cTglY(), track_0.cTglZ(),
                                       track_0.cTglSnp(), track_0.cTglTgl(),
                                       track_0.c1PtY(), track_0.c1PtZ(), track_0.c1PtSnp(),
                                       track_0.c1PtTgl(), track_0.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar0(x0_, alpha0_, arraypar0, covpar0);
      for (auto it1 = it0 + 1; it1 != tracks.end(); ++it1) {
        auto& track_1 = *it1;
        UChar_t clustermap_1 = track_1.itsClusterMap();
        bool isselected_1 = track_1.tpcNClsFound() > 70 && track_1.flags() & 0x4;
        isselected_1 = isselected_1 && (TESTBIT(clustermap_1, 0) || TESTBIT(clustermap_1, 1));
        if (!isselected_1)
          continue;
        if (track_0.signed1Pt() * track_1.signed1Pt() > 0)
          continue;
        float x1_ = track_1.x();
        float alpha1_ = track_1.alpha();
        std::array<float, 5> arraypar1 = {track_1.y(), track_1.z(), track_1.snp(),
                                          track_1.tgl(), track_1.signed1Pt()};
        std::array<float, 15> covpar1 = {track_1.cYY(), track_1.cZY(), track_1.cZZ(),
                                         track_1.cSnpY(), track_1.cSnpZ(),
                                         track_1.cSnpSnp(), track_1.cTglY(), track_1.cTglZ(),
                                         track_1.cTglSnp(), track_1.cTglTgl(),
                                         track_1.c1PtY(), track_1.c1PtZ(), track_1.c1PtSnp(),
                                         track_1.c1PtTgl(), track_1.c1Pt21Pt2()};
        o2::track::TrackParCov trackparvar1(x1_, alpha1_, arraypar1, covpar1);
        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar0, trackparvar1);
        if (nCand == 0)
          continue;
        const auto& vtx = df.getPCACandidate();
        LOGF(info, "vertex x %f", vtx[0]);
        hvtx_x_out->Fill(vtx[0]);
        hvtx_y_out->Fill(vtx[1]);
        hvtx_z_out->Fill(vtx[2]);
        o2::track::TrackParCov trackdec0 = df.getTrack(0);
        o2::track::TrackParCov trackdec1 = df.getTrack(1);
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        trackdec0.getPxPyPzGlo(pvec0);
        trackdec1.getPxPyPzGlo(pvec1);
        float masspion = 0.140;
        float masskaon = 0.494;
        float mass_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masspion,
                                     pvec1[0], pvec1[1], pvec1[2], masskaon);
        float masssw_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masskaon,
                                       pvec1[0], pvec1[1], pvec1[2], masspion);
        secvtx2prong(track_0.collisionId(),
                     collision.posX(), collision.posY(), collision.posZ(),
                     vtx[0], vtx[1], vtx[2], track_0.globalIndex(),
                     pvec0[0], pvec0[1], pvec0[2], track_0.y(),
                     track_1.globalIndex(), pvec1[0], pvec1[1], pvec1[2], track_1.y(),
                     mass_, masssw_);
        hchi2dca->Fill(df.getChi2AtPCACandidate());
      }
    }
  }
};

struct CandidateBuildingDzero {
  Produces<aod::Cand2Prong> cand2prong;
  void process(aod::SecVtx2Prong const& secVtx2Prongs,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    LOGF(info, "NEW EVENT CANDIDATE");

    o2::vertexing::DCAFitterN<2> df;
    df.setBz(5.0);
    // After finding the vertex, propagate tracks to the DCA. This is default anyway
    df.setPropagateToPCA(true);
    // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    df.setMaxR(200);
    // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    df.setMaxDZIni(4);
    // stop iterations if max correction is below this value. This is default anyway
    df.setMinParamChange(1e-3);
    // stop iterations if chi2 improves by less that this factor
    df.setMinRelChi2Change(0.9);

    for (auto& secVtx2prong : secVtx2Prongs) {
      LOGF(INFO, " ------- new event ---------");
      LOGF(INFO, " track0 y from secvtx tab.  %f", secVtx2prong.y0());
      LOGF(INFO, " track0 y from track  %f", secVtx2prong.index0().y());
      LOGF(INFO, " track1 y from secvtx table  %f", secVtx2prong.y1());
      LOGF(INFO, " track1 y from track  %f", secVtx2prong.index1().y());

      float x0_ = secVtx2prong.index0().x();
      float alpha0_ = secVtx2prong.index0().alpha();
      std::array<float, 5> arraypar0 = {secVtx2prong.index0().y(), secVtx2prong.index0().z(),
                                        secVtx2prong.index0().snp(), secVtx2prong.index0().tgl(),
                                        secVtx2prong.index0().signed1Pt()};
      std::array<float, 15> covpar0 = {secVtx2prong.index0().cYY(), secVtx2prong.index0().cZY(),
                                       secVtx2prong.index0().cZZ(), secVtx2prong.index0().cSnpY(),
                                       secVtx2prong.index0().cSnpZ(), secVtx2prong.index0().cSnpSnp(),
                                       secVtx2prong.index0().cTglY(), secVtx2prong.index0().cTglZ(),
                                       secVtx2prong.index0().cTglSnp(), secVtx2prong.index0().cTglTgl(),
                                       secVtx2prong.index0().c1PtY(), secVtx2prong.index0().c1PtZ(),
                                       secVtx2prong.index0().c1PtSnp(), secVtx2prong.index0().c1PtTgl(),
                                       secVtx2prong.index0().c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar0(x0_, alpha0_, arraypar0, covpar0);

      float x1_ = secVtx2prong.index1().x();
      float alpha1_ = secVtx2prong.index1().alpha();
      std::array<float, 5> arraypar1 = {secVtx2prong.index1().y(), secVtx2prong.index1().z(),
                                        secVtx2prong.index1().snp(), secVtx2prong.index1().tgl(),
                                        secVtx2prong.index1().signed1Pt()};
      std::array<float, 15> covpar1 = {secVtx2prong.index1().cYY(), secVtx2prong.index1().cZY(),
                                       secVtx2prong.index1().cZZ(), secVtx2prong.index1().cSnpY(),
                                       secVtx2prong.index1().cSnpZ(), secVtx2prong.index1().cSnpSnp(),
                                       secVtx2prong.index1().cTglY(), secVtx2prong.index1().cTglZ(),
                                       secVtx2prong.index1().cTglSnp(), secVtx2prong.index1().cTglTgl(),
                                       secVtx2prong.index1().c1PtY(), secVtx2prong.index1().c1PtZ(),
                                       secVtx2prong.index1().c1PtSnp(), secVtx2prong.index1().c1PtTgl(),
                                       secVtx2prong.index1().c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar1(x1_, alpha1_, arraypar1, covpar1);

      df.setUseAbsDCA(true);
      //FIXME: currently I rebuild the vertex for each track-track pair and
      //select the candidate via its index. It is redundant cause the secondary
      //vertex recostruction is performed more than once for each dca candidate
      int nCand = df.process(trackparvar0, trackparvar1);
      if (nCand == 0) {
        LOGF(error, " DCAFitter failing in the candidate building: it should not happen");
      }
      const auto& secvtx = df.getPCACandidate();
      float masspion = 0.140;
      float masskaon = 0.494;
      float mass_ = invmass2prongs(secVtx2prong.px0(), secVtx2prong.py0(),
                                   secVtx2prong.pz0(), masspion,
                                   secVtx2prong.px1(), secVtx2prong.py1(),
                                   secVtx2prong.pz1(), masskaon);
      float masssw_ = invmass2prongs(secVtx2prong.px0(), secVtx2prong.py0(),
                                     secVtx2prong.pz0(), masskaon,
                                     secVtx2prong.px1(), secVtx2prong.py1(),
                                     secVtx2prong.pz1(), masspion);
      cand2prong(mass_, masssw_);
      o2::track::TrackParCov trackdec0 = df.getTrack(0);
      o2::track::TrackParCov trackdec1 = df.getTrack(1);
      std::array<float, 3> pvec0;
      std::array<float, 3> pvec1;
      trackdec0.getPxPyPzGlo(pvec0);
      trackdec1.getPxPyPzGlo(pvec1);
      LOGF(info, "Pt track 0 from table %f and from calc %f", secVtx2prong.px0(), pvec0[0]);
      if (abs(secVtx2prong.px0() - pvec0[0]) > 0.000000001) {
        LOGF(info, "BIG ERRROR");
      }
    }
  }
};

struct DzeroHistoTask {
  // secondary vertex position
  OutputObj<TH1F> hvtx_x_outt{TH1F("hvtx_xt", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_outt{TH1F("hvtx_yt", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_outt{TH1F("hvtx_zt", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hmass_nocuts_out{TH1F("hmass_nocuts", "2-track inv mass", 500, 0, 5.0)};
  OutputObj<TH1F> hdecayxy{TH1F("hdecayxy", "decay length xy", 100, 0., 1.0)};
  OutputObj<TH1F> hdecayxyz{TH1F("hdecayxyz", "decay length", 100, 0., 1.0)};

  void process(soa::Join<aod::Cand2Prong, aod::SecVtx2Prong> const& secVtx2Prongs)
  {
    LOGF(info, "NEW EVENT");

    for (auto& secVtx2prong : secVtx2Prongs) {
      hvtx_y_outt->Fill(secVtx2prong.posdecayy());
      hvtx_z_outt->Fill(secVtx2prong.posdecayz());
      hvtx_x_outt->Fill(secVtx2prong.posdecayx());
      hvtx_y_outt->Fill(secVtx2prong.posdecayy());
      hvtx_z_outt->Fill(secVtx2prong.posdecayz());
      hdecayxy->Fill(secVtx2prong.decaylengthXY());
      hdecayxyz->Fill(secVtx2prong.decaylength());
      hmass_nocuts_out->Fill(secVtx2prong.mass());
      hmass_nocuts_out->Fill(secVtx2prong.massbar());
      LOGF(info, "new event");
      LOGF(info, "mass %f", secVtx2prong.mass());
      LOGF(info, "mass from cand %f", secVtx2prong.massD0());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<DecayVertexBuilder2Prong>("vertexerhf-decayvertexbuilder2prong"),
    adaptAnalysisTask<CandidateBuildingDzero>("vertexerhf-candidatebuildingDzero"),
    adaptAnalysisTask<DzeroHistoTask>("vertexerhf-Dzerotask")};
}
