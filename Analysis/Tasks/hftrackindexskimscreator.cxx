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
#include "Analysis/trackUtilities.h"
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

/// Track selection
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
      int status = 1; // selection flag
      if (b_dovalplots)
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
      if (b_dovalplots) {
        if (status == 1) {
          hpt_cuts->Fill(track_0.pt());
          hdcatoprimxy_cuts->Fill(dca[0]);
        }
      }
      seltrack(status, dca[0], dca[1]);
    }
  }
};

/// Pre-selection of 2-prong and 3-prong secondary vertices
struct HFTrackIndexSkimsCreator {
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
  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);

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

    double mass2PiK{0};
    double mass2KPi{0};
    double mass3PiKPi{0};

    // first loop over positive tracks
    for (auto i_p1 = tracks.begin(); i_p1 != tracks.end(); ++i_p1) {
      auto& track_p1 = *i_p1;
      if (track_p1.signed1Pt() < 0)
        continue;
      auto trackparvar_p1 = getTrackParCov(track_p1);

      // first loop over negative tracks
      for (auto i_n1 = tracks.begin(); i_n1 != tracks.end(); ++i_n1) {
        auto& track_n1 = *i_n1;
        if (track_n1.signed1Pt() > 0)
          continue;
        auto trackparvar_n1 = getTrackParCov(track_n1);

        // reconstruct the 2-prong secondary vertex
        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar_p1, trackparvar_n1);
        if (nCand == 0)
          continue;
        const auto& vtx = df.getPCACandidate();
        array<float, 3> pvec0;
        array<float, 3> pvec1;
        df.getTrack(0).getPxPyPzGlo(pvec0);
        df.getTrack(1).getPxPyPzGlo(pvec1);

        auto arrMom = array{pvec0, pvec1};
        mass2PiK = RecoDecay::M(arrMom, array{massPi, massK});
        mass2KPi = RecoDecay::M(arrMom, array{massK, massPi});

        if (b_dovalplots) {
          hmass2->Fill(mass2PiK);
          hmass2->Fill(mass2KPi);
        }

        hftrackindexprong2(track_p1.collisionId(),
                           track_p1.globalIndex(),
                           track_n1.globalIndex(), 1);

        // 3-prong vertex reconstruction
        if (do3prong == 1) {
          // second loop over positive tracks
          for (auto i_p2 = i_p1 + 1; i_p2 != tracks.end(); ++i_p2) {
            auto& track_p2 = *i_p2;
            if (track_p2.signed1Pt() < 0)
              continue;

            auto arr3Mom = array{
              array{track_p1.px(), track_p1.py(), track_p1.pz()},
              array{track_n1.px(), track_n1.py(), track_n1.pz()},
              array{track_p2.px(), track_p2.py(), track_p2.pz()}};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (mass3PiKPi < d_minmassDp || mass3PiKPi > d_maxmassDp)
              continue;

            auto trackparvar_p2 = getTrackParCov(track_p2);
            df3.setUseAbsDCA(true);
            int nCand3 = df3.process(trackparvar_p1, trackparvar_n1, trackparvar_p2);
            if (nCand3 == 0)
              continue;
            const auto& vtx3 = df3.getPCACandidate();
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);

            arr3Mom = array{pvec0, pvec1, pvec2};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (b_dovalplots) {
              hmass3->Fill(mass3PiKPi);
            }

            hftrackindexprong3(track_p1.collisionId(),
                               track_p1.globalIndex(),
                               track_n1.globalIndex(),
                               track_p2.globalIndex(), 2);
          }
          // second loop over negative tracks
          for (auto i_n2 = i_n1 + 1; i_n2 != tracks.end(); ++i_n2) {
            auto& track_n2 = *i_n2;
            if (track_n2.signed1Pt() > 0)
              continue;

            auto arr3Mom = array{
              array{track_n1.px(), track_n1.py(), track_n1.pz()},
              array{track_p1.px(), track_p1.py(), track_p1.pz()},
              array{track_n2.px(), track_n2.py(), track_n2.pz()}};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (mass3PiKPi < d_minmassDp || mass3PiKPi > d_maxmassDp)
              continue;

            auto trackparvar_n2 = getTrackParCov(track_n2);
            df3.setUseAbsDCA(true);
            int nCand3 = df3.process(trackparvar_n1, trackparvar_p1, trackparvar_n2);
            if (nCand3 == 0)
              continue;
            const auto& vtx3 = df3.getPCACandidate();
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);

            arr3Mom = array{pvec0, pvec1, pvec2};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (b_dovalplots) {
              hmass3->Fill(mass3PiKPi);
            }

            hftrackindexprong3(track_n1.collisionId(),
                               track_n1.globalIndex(),
                               track_p1.globalIndex(),
                               track_n2.globalIndex(), 2);
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
