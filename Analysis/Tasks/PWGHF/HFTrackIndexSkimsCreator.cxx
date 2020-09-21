// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file hftrackindexskimscreator.cxx
/// \brief Pre-selection of 2-prong and 3-prong secondary vertices of heavy-flavour decay candidates
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "Analysis/SecondaryVertexHF.h"
#include "Analysis/trackUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

/// Track selection
struct SelectTracks {
  Produces<aod::SelTrack> rowSelectedTrack;
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
    for (auto& track : tracks) {
      int status = 1; // selection flag
      if (b_dovalplots)
        hpt_nocuts->Fill(track.pt());
      if (track.pt() < ptmintrack)
        status = 0;
      UChar_t clustermap_0 = track.itsClusterMap();
      bool isselected_0 = track.tpcNClsFound() >= d_tpcnclsfound && track.flags() & 0x4;
      isselected_0 = isselected_0 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        status = 0;
      array<float, 2> dca;
      auto trackparvar0 = getTrackParCov(track);
      bool isprop = trackparvar0.propagateParamToDCA(vtxXYZ, d_bz, &dca, 100.); // get impact parameters
      if (!isprop)
        status = 0;
      if (abs(dca[0]) < dcatoprimxymin)
        status = 0;
      if (b_dovalplots) {
        if (status == 1) {
          hpt_cuts->Fill(track.pt());
          hdcatoprimxy_cuts->Fill(dca[0]);
        }
      }
      rowSelectedTrack(status, dca[0], dca[1]);
    }
  }
};

/// Pre-selection of 2-prong and 3-prong secondary vertices
struct HFTrackIndexSkimsCreator {
  Produces<aod::HfTrackIndexProng2> rowTrackIndexProng2;
  Produces<aod::HfTrackIndexProng3> rowTrackIndexProng3;
  Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};
  Configurable<int> do3prong{"do3prong", 0, "do 3 prong"};
  Configurable<double> d_bz{"d_bz", 5.0, "bz field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200, "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4, "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations is chi2/chi2old > this"};
  Configurable<double> d_minmassDp{"d_minmassDp", 1.5, "min mass dplus presel"};
  Configurable<double> d_maxmassDp{"d_maxmassDp", 2.1, "max mass dplus presel"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};

  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 1000, -2.0, 2.0)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 1000, -2.0, 2.0)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 1000, -20.0, 20.0)};
  OutputObj<TH1F> hmass2{TH1F("hmass2", "; Inv Mass (GeV/c^{2})", 500, 0, 5.0)};
  OutputObj<TH1F> hmass3{TH1F("hmass3", "; Inv Mass (GeV/c^{2})", 500, 1.6, 2.1)};
  OutputObj<TH1F> hvtx3_x_out{TH1F("hvtx3_x", "3-track vtx", 1000, -2.0, 2.0)};
  OutputObj<TH1F> hvtx3_y_out{TH1F("hvtx3_y", "3-track vtx", 1000, -2.0, 2.0)};
  OutputObj<TH1F> hvtx3_z_out{TH1F("hvtx3_z", "3-track vtx", 1000, -20.0, 20.0)};

  Filter filterSelectTracks = aod::seltrack::issel == 1;
  using SelectedTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::SelTrack>>;
  // FIXME
  //Partition<SelectedTracks> tracksPos = aod::track::signed1Pt > 0.f;
  //Partition<SelectedTracks> tracksNeg = aod::track::signed1Pt < 0.f;

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double mass2PiK{0};
  double mass2KPi{0};
  double mass3PiKPi{0};

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               SelectedTracks const& tracks)
  {
    int trigindex = int{triggerindex};
    if (trigindex != -1) {
      uint64_t triggerMask = collision.bc().triggerMask();
      bool isTriggerClassFired = triggerMask & 1ul << (trigindex - 1);
      if (!isTriggerClassFired)
        return;
    }

    // 2-prong vertex fitter
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(d_bz);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);
    df.setUseAbsDCA(true);

    // 3-prong vertex fitter
    o2::vertexing::DCAFitterN<3> df3;
    df3.setBz(d_bz);
    df3.setPropagateToPCA(b_propdca);
    df3.setMaxR(d_maxr);
    df3.setMaxDZIni(d_maxdzini);
    df3.setMinParamChange(d_minparamchange);
    df3.setMinRelChi2Change(d_minrelchi2change);
    df3.setUseAbsDCA(true);

    // first loop over positive tracks
    //for (auto trackPos1 = tracksPos.begin(); trackPos1 != tracksPos.end(); ++trackPos1) {
    for (auto trackPos1 = tracks.begin(); trackPos1 != tracks.end(); ++trackPos1) {
      if (trackPos1.signed1Pt() < 0)
        continue;

      auto trackParVarPos1 = getTrackParCov(trackPos1);

      // first loop over negative tracks
      //for (auto trackNeg1 = tracksNeg.begin(); trackNeg1 != tracksNeg.end(); ++trackNeg1) {
      for (auto trackNeg1 = tracks.begin(); trackNeg1 != tracks.end(); ++trackNeg1) {
        if (trackNeg1.signed1Pt() > 0)
          continue;

        auto trackParVarNeg1 = getTrackParCov(trackNeg1);

        // reconstruct the 2-prong secondary vertex
        if (df.process(trackParVarPos1, trackParVarNeg1) == 0)
          continue;

        // get vertex
        //const auto& vtx = df.getPCACandidate();

        // get track momenta
        array<float, 3> pvec0;
        array<float, 3> pvec1;
        df.getTrack(0).getPxPyPzGlo(pvec0);
        df.getTrack(1).getPxPyPzGlo(pvec1);
        const auto& secondaryVertex = df.getPCACandidate();

        // calculate invariant masses
        auto arrMom = array{pvec0, pvec1};
        mass2PiK = RecoDecay::M(arrMom, array{massPi, massK});
        mass2KPi = RecoDecay::M(arrMom, array{massK, massPi});

        if (b_dovalplots) {
          hmass2->Fill(mass2PiK);
          hmass2->Fill(mass2KPi);
          hvtx_x_out->Fill(secondaryVertex[0]);
          hvtx_y_out->Fill(secondaryVertex[1]);
          hvtx_z_out->Fill(secondaryVertex[2]);
        }

        // fill table row
        rowTrackIndexProng2(trackPos1.collisionId(),
                            trackPos1.globalIndex(),
                            trackNeg1.globalIndex(), 1);

        // 3-prong vertex reconstruction
        if (do3prong == 1) {
          // second loop over positive tracks
          //for (auto trackPos2 = trackPos1 + 1; trackPos2 != tracksPos.end(); ++trackPos2) {
          for (auto trackPos2 = trackPos1 + 1; trackPos2 != tracks.end(); ++trackPos2) {
            if (trackPos2.signed1Pt() < 0)
              continue;

            // calculate invariant mass
            auto arr3Mom = array{
              array{trackPos1.px(), trackPos1.py(), trackPos1.pz()},
              array{trackNeg1.px(), trackNeg1.py(), trackNeg1.pz()},
              array{trackPos2.px(), trackPos2.py(), trackPos2.pz()}};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (mass3PiKPi < d_minmassDp || mass3PiKPi > d_maxmassDp)
              continue;

            auto trackParVarPos2 = getTrackParCov(trackPos2);

            // reconstruct the 3-prong secondary vertex
            if (df3.process(trackParVarPos1, trackParVarNeg1, trackParVarPos2) == 0)
              continue;

            // get vertex
            //const auto& vtx3 = df3.getPCACandidate();

            // get track momenta
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);
            const auto& secondaryVertex3 = df3.getPCACandidate();

            // calculate invariant mass
            arr3Mom = array{pvec0, pvec1, pvec2};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (b_dovalplots) {
              hmass3->Fill(mass3PiKPi);
              hvtx3_x_out->Fill(secondaryVertex3[0]);
              hvtx3_y_out->Fill(secondaryVertex3[1]);
              hvtx3_z_out->Fill(secondaryVertex3[2]);
            }

            // fill table row
            rowTrackIndexProng3(trackPos1.collisionId(),
                                trackPos1.globalIndex(),
                                trackNeg1.globalIndex(),
                                trackPos2.globalIndex(), 2);
          }
          // second loop over negative tracks
          //for (auto trackNeg2 = trackNeg1 + 1; trackNeg2 != tracksNeg.end(); ++trackNeg2) {
          for (auto trackNeg2 = trackNeg1 + 1; trackNeg2 != tracks.end(); ++trackNeg2) {
            if (trackNeg2.signed1Pt() > 0)
              continue;

            // calculate invariant mass
            auto arr3Mom = array{
              array{trackNeg1.px(), trackNeg1.py(), trackNeg1.pz()},
              array{trackPos1.px(), trackPos1.py(), trackPos1.pz()},
              array{trackNeg2.px(), trackNeg2.py(), trackNeg2.pz()}};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (mass3PiKPi < d_minmassDp || mass3PiKPi > d_maxmassDp)
              continue;

            auto trackParVarNeg2 = getTrackParCov(trackNeg2);

            // reconstruct the 3-prong secondary vertex
            if (df3.process(trackParVarNeg1, trackParVarPos1, trackParVarNeg2) == 0)
              continue;

            // get vertex
            //const auto& vtx3 = df3.getPCACandidate();

            // get track momenta
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);

            // calculate invariant mass
            arr3Mom = array{pvec0, pvec1, pvec2};
            mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});

            if (b_dovalplots) {
              hmass3->Fill(mass3PiKPi);
            }

            // fill table row
            rowTrackIndexProng3(trackNeg1.collisionId(),
                                trackNeg1.globalIndex(),
                                trackPos1.globalIndex(),
                                trackNeg2.globalIndex(), 2);
          }
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<SelectTracks>("hf-produce-sel-track"),
    adaptAnalysisTask<HFTrackIndexSkimsCreator>("hf-track-index-skims-creator")};
}
