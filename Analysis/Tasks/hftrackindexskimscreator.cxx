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
#include "Analysis/SecondaryVertexHF.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Analysis/RecoDecay.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

namespace o2::aod
{
namespace seltrack
{
DECLARE_SOA_COLUMN(IsSel, issel, int);
DECLARE_SOA_COLUMN(DCAPrim0, dcaprim0, float);
DECLARE_SOA_COLUMN(DCAPrim1, dcaprim1, float);
} // namespace seltrack
DECLARE_SOA_TABLE(SelTrack, "AOD", "SELTRACK", seltrack::IsSel, seltrack::DCAPrim0,
                  seltrack::DCAPrim1);
} // namespace o2::aod

struct SelectTracks {
  Produces<aod::SelTrack> seltrack;
  Configurable<double> ptmintrack{"ptmintrack", -1, "ptmin single track"};
  Configurable<double> dcatoprimxymin{"dcatoprimxymin", 0, "dca xy to prim vtx min"};
  Configurable<int> d_tpcnclsfound{"d_tpcnclsfound", 70, "min number of tpc cls >="};
  Configurable<double> d_bz{"d_bz", 5.0, "bz field"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};
  OutputObj<TH1F> hpt_nocuts{TH1F("hpt_nocuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hpt_cuts{TH1F("hpt_cuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hdcatoprimxy_cuts{TH1F("hdcatoprimxy_cuts", "dca xy to prim. vertex (cm)", 100, -1.0, 1.0)};

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    Point3D<float> vtxXYZ(collision.posX(), collision.posY(), collision.posZ());
    for (auto it0 = tracks.begin(); it0 != tracks.end(); ++it0) {
      auto& track_0 = *it0;
      int status = 1;
      if (b_dovalplots == true)
        hpt_nocuts->Fill(track_0.pt());
      if (track_0.pt() < ptmintrack)
        status = 0;
      UChar_t clustermap_0 = track_0.itsClusterMap();
      bool isselected_0 = track_0.tpcNClsFound() >= d_tpcnclsfound && track_0.flags() & 0x4;
      isselected_0 = isselected_0 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        status = 0;
      array<float, 2> dca;
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
      trackparvar0.propagateParamToDCA(vtxXYZ, d_bz, &dca);
      if (abs(dca[0]) < dcatoprimxymin)
        status = 0;
      if (b_dovalplots == true) {
        if (status == 1) {
          hpt_cuts->Fill(track_0.pt());
          hdcatoprimxy_cuts->Fill(dca[0]);
        }
      }
      seltrack(status, dca[0], dca[1]);
    }
  }
};

struct HFTrackIndexSkimsCreator {
  float masspion = 0.140;
  float masskaon = 0.494;
  OutputObj<TH1F> hmass2{TH1F("hmass2", "; Inv Mass (GeV/c^{2})", 500, 0, 5.0)};
  OutputObj<TH1F> hmass3{TH1F("hmass3", "; Inv Mass (GeV/c^{2})", 500, 0, 5.0)};
  Produces<aod::HfTrackIndexProng2> hftrackindexprong2;
  Produces<aod::HfTrackIndexProng3> hftrackindexprong3;
  Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};
  Configurable<int> do3prong{"do3prong", 0, "do 3 prong"};
  Configurable<double> d_bz{"d_bz", 5.0, "bz field"};
  Configurable<bool> b_propdca{"b_propdca", true,
                               "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200, "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4,
                                  "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1e-3,
                                        "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9,
                                          "stop iterations is chi2/chi2old > this"};
  Configurable<double> d_minmassDp{"d_minmassDp", 1.5, "min mass dplus presel"};
  Configurable<double> d_maxmassDp{"d_maxmassDp", 2.1, "max mass dplus presel"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};
  Filter seltrack = (aod::seltrack::issel == 1);

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               soa::Filtered<soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::SelTrack>> const& tracks)
  {
    int trigindex = int{triggerindex};
    if (trigindex != -1) {
      //LOGF(info, "Selecting on trigger bit %d", trigindex);
      uint64_t triggerMask = collision.bc().triggerMask();
      bool isTriggerClassFired = triggerMask & 1ul << (trigindex - 1);
      if (!isTriggerClassFired)
        return;
    }

    LOGF(info, "N. of Tracks for collision: %d", tracks.size());
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(d_bz);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);

    o2::vertexing::DCAFitterN<3> df3;
    df3.setBz(d_bz);
    df3.setPropagateToPCA(b_propdca);
    df3.setMaxR(d_maxr);
    df3.setMaxDZIni(d_maxdzini);
    df3.setMinParamChange(d_minparamchange);
    df3.setMinRelChi2Change(d_minrelchi2change);

    for (auto i_p1 = tracks.begin(); i_p1 != tracks.end(); ++i_p1) {
      auto& track_p1 = *i_p1;
      if (track_p1.signed1Pt() < 0)
        continue;
      float x_p1 = track_p1.x();
      float alpha_p1 = track_p1.alpha();
      std::array<float, 5> arraypar_p1 = {track_p1.y(), track_p1.z(), track_p1.snp(),
                                          track_p1.tgl(), track_p1.signed1Pt()};
      std::array<float, 15> covpar_p1 = {track_p1.cYY(), track_p1.cZY(), track_p1.cZZ(),
                                         track_p1.cSnpY(), track_p1.cSnpZ(),
                                         track_p1.cSnpSnp(), track_p1.cTglY(), track_p1.cTglZ(),
                                         track_p1.cTglSnp(), track_p1.cTglTgl(),
                                         track_p1.c1PtY(), track_p1.c1PtZ(), track_p1.c1PtSnp(),
                                         track_p1.c1PtTgl(), track_p1.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar_p1(x_p1, alpha_p1, arraypar_p1, covpar_p1);
      for (auto i_n1 = tracks.begin(); i_n1 != tracks.end(); ++i_n1) {
        auto& track_n1 = *i_n1;
        if (track_n1.signed1Pt() > 0)
          continue;
        float x_n1 = track_n1.x();
        float alpha_n1 = track_n1.alpha();
        std::array<float, 5> arraypar_n1 = {track_n1.y(), track_n1.z(), track_n1.snp(),
                                            track_n1.tgl(), track_n1.signed1Pt()};
        std::array<float, 15> covpar_n1 = {track_n1.cYY(), track_n1.cZY(), track_n1.cZZ(),
                                           track_n1.cSnpY(), track_n1.cSnpZ(),
                                           track_n1.cSnpSnp(), track_n1.cTglY(), track_n1.cTglZ(),
                                           track_n1.cTglSnp(), track_n1.cTglTgl(),
                                           track_n1.c1PtY(), track_n1.c1PtZ(), track_n1.c1PtSnp(),
                                           track_n1.c1PtTgl(), track_n1.c1Pt21Pt2()};
        o2::track::TrackParCov trackparvar_n1(x_n1, alpha_n1, arraypar_n1, covpar_n1);
        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar_p1, trackparvar_n1);
        if (nCand == 0)
          continue;
        const auto& vtx = df.getPCACandidate();
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        df.getTrack(0).getPxPyPzGlo(pvec0);
        df.getTrack(1).getPxPyPzGlo(pvec1);
        float mass_ = sqrt(invmass2prongs2(pvec0[0], pvec0[1],
                                           pvec0[2], masspion,
                                           pvec1[0], pvec1[1],
                                           pvec1[2], masskaon));
        float masssw_ = sqrt(invmass2prongs2(pvec0[0], pvec0[1],
                                             pvec0[2], masskaon,
                                             pvec1[0], pvec1[1],
                                             pvec1[2], masspion));
        if (b_dovalplots == true) {
          hmass2->Fill(mass_);
          hmass2->Fill(masssw_);
        }
        hftrackindexprong2(track_p1.collisionId(),
                           track_p1.globalIndex(),
                           track_n1.globalIndex(), 1);
        if (do3prong == 1) {
          //second loop on positive tracks
          for (auto i_p2 = i_p1 + 1; i_p2 != tracks.end(); ++i_p2) {
            auto& track_p2 = *i_p2;
            if (track_p2.signed1Pt() < 0)
              continue;
            float x_p2 = track_p2.x();
            float alpha_p2 = track_p2.alpha();
            double mass3prong2 = invmass3prongs2(track_p1.px(), track_p1.py(), track_p1.pz(), masspion,
                                                 track_n1.px(), track_n1.py(), track_n1.pz(), masskaon,
                                                 track_p2.px(), track_p2.py(), track_p2.pz(), masspion);
            if (mass3prong2 < d_minmassDp * d_minmassDp || mass3prong2 > d_maxmassDp * d_maxmassDp)
              continue;
            if (b_dovalplots == true)
              hmass3->Fill(sqrt(mass3prong2));
            std::array<float, 5> arraypar_p2 = {track_p2.y(), track_p2.z(), track_p2.snp(),
                                                track_p2.tgl(), track_p2.signed1Pt()};
            std::array<float, 15> covpar_p2 = {track_p2.cYY(), track_p2.cZY(), track_p2.cZZ(),
                                               track_p2.cSnpY(), track_p2.cSnpZ(),
                                               track_p2.cSnpSnp(), track_p2.cTglY(), track_p2.cTglZ(),
                                               track_p2.cTglSnp(), track_p2.cTglTgl(),
                                               track_p2.c1PtY(), track_p2.c1PtZ(), track_p2.c1PtSnp(),
                                               track_p2.c1PtTgl(), track_p2.c1Pt21Pt2()};
            o2::track::TrackParCov trackparvar_p2(x_p2, alpha_p2, arraypar_p2, covpar_p2);
            df3.setUseAbsDCA(true);
            int nCand3 = df3.process(trackparvar_p1, trackparvar_n1, trackparvar_p2);
            if (nCand3 == 0)
              continue;
            const auto& vtx3 = df3.getPCACandidate();
            std::array<float, 3> pvec0;
            std::array<float, 3> pvec1;
            std::array<float, 3> pvec2;
            df.getTrack(0).getPxPyPzGlo(pvec0);
            df.getTrack(1).getPxPyPzGlo(pvec1);
            df.getTrack(2).getPxPyPzGlo(pvec2);
            float mass_ = sqrt(invmass3prongs2(pvec0[0], pvec0[1],
                                               pvec0[2], masspion,
                                               pvec1[0], pvec1[1],
                                               pvec1[2], masskaon,
                                               pvec2[0], pvec2[1],
                                               pvec2[2], masspion));
            if (b_dovalplots == true) {
              hmass3->Fill(mass_);
            }
            hftrackindexprong3(track_p1.collisionId(),
                               track_p1.globalIndex(),
                               track_n1.globalIndex(),
                               track_p1.globalIndex(), 2);
          }
          //second loop on negative tracks
          for (auto i_n2 = i_n1 + 1; i_n2 != tracks.end(); ++i_n2) {
            auto& track_n2 = *i_n2;
            if (track_n2.signed1Pt() > 0)
              continue;
            float x_n2 = track_n2.x();
            float alpha_n2 = track_n2.alpha();
            double mass3prong2 = invmass3prongs2(track_n1.px(), track_n1.py(), track_n1.pz(), masspion,
                                                 track_p1.px(), track_p1.py(), track_p1.pz(), masskaon,
                                                 track_n2.px(), track_n2.py(), track_n2.pz(), masspion);
            if (mass3prong2 < d_minmassDp * d_minmassDp || mass3prong2 > d_maxmassDp * d_maxmassDp)
              continue;
            hmass3->Fill(sqrt(mass3prong2));
            std::array<float, 5> arraypar_n2 = {track_n2.y(), track_n2.z(), track_n2.snp(),
                                                track_n2.tgl(), track_n2.signed1Pt()};
            std::array<float, 15> covpar_n2 = {track_n2.cYY(), track_n2.cZY(), track_n2.cZZ(),
                                               track_n2.cSnpY(), track_n2.cSnpZ(),
                                               track_n2.cSnpSnp(), track_n2.cTglY(), track_n2.cTglZ(),
                                               track_n2.cTglSnp(), track_n2.cTglTgl(),
                                               track_n2.c1PtY(), track_n2.c1PtZ(), track_n2.c1PtSnp(),
                                               track_n2.c1PtTgl(), track_n2.c1Pt21Pt2()};
            o2::track::TrackParCov trackparvar_n2(x_n2, alpha_n2, arraypar_n2, covpar_n2);
            df3.setUseAbsDCA(true);
            int nCand3 = df3.process(trackparvar_n1, trackparvar_p1, trackparvar_n2);
            if (nCand3 == 0)
              continue;
            const auto& vtx3 = df3.getPCACandidate();
            std::array<float, 3> pvec0;
            std::array<float, 3> pvec1;
            std::array<float, 3> pvec2;
            df.getTrack(0).getPxPyPzGlo(pvec0);
            df.getTrack(1).getPxPyPzGlo(pvec1);
            df.getTrack(2).getPxPyPzGlo(pvec2);
            float mass_ = sqrt(invmass3prongs2(pvec0[0], pvec0[1],
                                               pvec0[2], masspion,
                                               pvec1[0], pvec1[1],
                                               pvec1[2], masskaon,
                                               pvec2[0], pvec2[1],
                                               pvec2[2], masspion));
            if (b_dovalplots == true) {
              hmass3->Fill(mass_);
            }
            hftrackindexprong3(track_n1.collisionId(),
                               track_n1.globalIndex(),
                               track_p1.globalIndex(),
                               track_n1.globalIndex(), 2.);
          }
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<SelectTracks>("produce-sel-track"),
    adaptAnalysisTask<HFTrackIndexSkimsCreator>("vertexerhf-hftrackindexskimscreator")};
}
