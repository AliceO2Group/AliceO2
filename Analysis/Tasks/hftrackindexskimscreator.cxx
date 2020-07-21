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
#include "PID/PIDResponse.h"
#include "Analysis/trackUtilities.h"

#include <TFile.h>
#include <TH1F.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;
using namespace ROOT::Math;

namespace o2::aod
{
namespace hfseltrack
{
enum Select {
  IsTrackSel = BIT(0),
  IsPion = BIT(1),
  IsKaon = BIT(2),
  IsProton = BIT(3)
};

DECLARE_SOA_COLUMN(IsSel, issel, int);
DECLARE_SOA_COLUMN(DCAPrim0, dcaprim0, float);
DECLARE_SOA_COLUMN(DCAPrim1, dcaprim1, float);
} // namespace hfseltrack
DECLARE_SOA_TABLE(SelTrack, "AOD", "SELTRACK", hfseltrack::IsSel, hfseltrack::DCAPrim0,
                  hfseltrack::DCAPrim1);
} // namespace o2::aod

struct SelectTracks {
  Produces<aod::SelTrack> hfseltrack;
  Configurable<double> ptmintrack{"ptmintrack", -1, "ptmin single track"};
  Configurable<double> dcatoprimxymin{"dcatoprimxymin", 0, "dca xy to prim vtx min"};
  Configurable<int> d_tpcnclsfound{"d_tpcnclsfound", 70, "min number of tpc cls >="};
  Configurable<double> d_bz{"d_bz", 5.0, "bz field"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};
  OutputObj<TH1F> hpt_nocuts{TH1F("hpt_nocuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hpt_cuts{TH1F("hpt_cuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hdcatoprimxy_cuts{TH1F("hdcatoprimxy_cuts", "dca xy to prim. vertex (cm)", 100, -1.0, 1.0)};

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::pidRespTOF> const& tracks)
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
      auto trackparvar0 = getTrackParCov(track_0);
      trackparvar0.propagateParamToDCA(vtxXYZ, d_bz, &dca);
      if (abs(dca[0]) < dcatoprimxymin)
        status = 0;
      if (status) {
        if (TMath::Abs(track_0.nSigmaPi()) < 3)
          status |= BIT(o2::aod::hfseltrack::IsPion);
        if (TMath::Abs(track_0.nSigmaKa()) < 3)
          status |= BIT(o2::aod::hfseltrack::IsKaon);
        if (TMath::Abs(track_0.nSigmaPr()) < 3)
          status |= BIT(o2::aod::hfseltrack::IsProton);
      }
      if (b_dovalplots == true) {
        if (status) {
          hpt_cuts->Fill(track_0.pt());
          hdcatoprimxy_cuts->Fill(dca[0]);
        }
      }
      hfseltrack(status, dca[0], dca[1]);
    }
  }
};

struct HFTrackIndexSkimsCreator {
  std::vector<PxPyPzMVector> listTracks;
  double masspion = TDatabasePDG::Instance()->GetParticle(kPiPlus)->Mass();
  double masskaon = TDatabasePDG::Instance()->GetParticle(kKPlus)->Mass();
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
  Filter hfseltrack = (aod::hfseltrack::issel == 1);

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
      auto trackparvar_p1 = getTrackParCov(track_p1);
      for (auto i_n1 = tracks.begin(); i_n1 != tracks.end(); ++i_n1) {
        auto& track_n1 = *i_n1;
        if (track_n1.signed1Pt() > 0)
          continue;
        auto trackparvar_n1 = getTrackParCov(track_n1);
        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar_p1, trackparvar_n1);
        if (nCand == 0)
          continue;
        const auto& vtx = df.getPCACandidate();
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        df.getTrack(0).getPxPyPzGlo(pvec0);
        df.getTrack(1).getPxPyPzGlo(pvec1);

        addTrack(listTracks, pvec0, masspion);
        addTrack(listTracks, pvec1, masskaon);
        double mass_ = (sumOfTracks(listTracks)).M();
        listTracks[0].SetM(masskaon);
        listTracks[1].SetM(masspion);
        double masssw_ = (sumOfTracks(listTracks)).M();
        listTracks.clear();

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

            addTrack(listTracks, track_p1.px(), track_p1.py(), track_p1.pz(), masspion);
            addTrack(listTracks, track_n1.px(), track_n1.py(), track_n1.pz(), masskaon);
            addTrack(listTracks, track_p2.px(), track_p2.py(), track_p2.pz(), masspion);
            double mass3prong = (sumOfTracks(listTracks)).M();
            listTracks.clear();

            if (mass3prong < d_minmassDp || mass3prong > d_maxmassDp)
              continue;
            if (b_dovalplots == true)
              hmass3->Fill(mass3prong);
            auto trackparvar_p2 = getTrackParCov(track_p2);
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

            addTrack(listTracks, pvec0, masspion);
            addTrack(listTracks, pvec1, masskaon);
            addTrack(listTracks, pvec2, masspion);
            double mass_ = (sumOfTracks(listTracks)).M();
            listTracks.clear();

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

            addTrack(listTracks, track_n1.px(), track_n1.py(), track_n1.pz(), masspion);
            addTrack(listTracks, track_p1.px(), track_p1.py(), track_p1.pz(), masskaon);
            addTrack(listTracks, track_n2.px(), track_n2.py(), track_n2.pz(), masspion);
            double mass3prong = (sumOfTracks(listTracks)).M();
            listTracks.clear();

            if (mass3prong < d_minmassDp || mass3prong > d_maxmassDp)
              continue;
            hmass3->Fill(mass3prong);
            auto trackparvar_n2 = getTrackParCov(track_n2);
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

            addTrack(listTracks, pvec0, masspion);
            addTrack(listTracks, pvec1, masskaon);
            addTrack(listTracks, pvec2, masspion);
            double mass_ = (sumOfTracks(listTracks)).M();
            listTracks.clear();

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
