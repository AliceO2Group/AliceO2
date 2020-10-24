// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFTrackIndexSkimsCreator.cxx
/// \brief Pre-selection of 2-prong and 3-prong secondary vertices of heavy-flavour decay candidates
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "Analysis/HFSecondaryVertex.h"
#include "Analysis/trackUtilities.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

/// Track selection
struct SelectTracks {
  Produces<aod::HFSelTrack> rowSelectedTrack;

  Configurable<bool> b_dovalplots{"b_dovalplots", true, "fill histograms"};
  Configurable<double> d_bz{"d_bz", 5., "bz field"};
  // quality cut
  Configurable<bool> doCutQuality{"doCutQuality", true, "apply quality cuts"};
  Configurable<int> d_tpcnclsfound{"d_tpcnclsfound", 70, "min. number of TPC clusters"};
  // 2-prong cuts
  Configurable<double> ptmintrack_2prong{"ptmintrack_2prong", -1., "min. track pT"};
  Configurable<double> dcatoprimxymin_2prong{"dcatoprimxymin_2prong", 0., "min. DCAXY to prim. vtx."};
  Configurable<double> etamax_2prong{"etamax_2prong", 999., "max. pseudorapidity"};
  // 3-prong cuts
  Configurable<double> ptmintrack_3prong{"ptmintrack_3prong", -1., "min. track pT"};
  Configurable<double> dcatoprimxymin_3prong{"dcatoprimxymin_3prong", 0., "min. DCAXY to prim. vtx."};
  Configurable<double> etamax_3prong{"etamax_3prong", 999., "max. pseudorapidity"};

  OutputObj<TH1F> hpt_nocuts{TH1F("hpt_nocuts", "all tracks;#it{p}_{T}^{track} (GeV/#it{c});entries", 100, 0., 10.)};
  // 2-prong histograms
  OutputObj<TH1F> hpt_cuts_2prong{TH1F("hpt_cuts_2prong", "tracks selected for 2-prong vertexing;#it{p}_{T}^{track} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hdcatoprimxy_cuts_2prong{TH1F("hdcatoprimxy_cuts_2prong", "tracks selected for 2-prong vertexing;DCAxy to prim. vtx. (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> heta_cuts_2prong{TH1F("heta_cuts_2prong", "tracks selected for 2-prong vertexing;#it{#eta};entries", 100, -1., 1.)};
  // 3-prong histograms
  OutputObj<TH1F> hpt_cuts_3prong{TH1F("hpt_cuts_3prong", "tracks selected for 3-prong vertexing;#it{p}_{T}^{track} (GeV/#it{c});entries", 100, 0., 10.)};
  OutputObj<TH1F> hdcatoprimxy_cuts_3prong{TH1F("hdcatoprimxy_cuts_3prong", "tracks selected for 3-prong vertexing;DCAxy to prim. vtx. (cm);entries", 100, -1., 1.)};
  OutputObj<TH1F> heta_cuts_3prong{TH1F("heta_cuts_3prong", "tracks selected for 3-prong vertexing;#it{#eta};entries", 100, -1., 1.)};

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    math_utils::Point3D<float> vtxXYZ(collision.posX(), collision.posY(), collision.posZ());
    for (auto& track : tracks) {
      int status_2prong = 1; // selection flag
      int status_3prong = 1; // selection flag

      if (b_dovalplots) {
        hpt_nocuts->Fill(track.pt());
      }

      // pT cut
      if (track.pt() < ptmintrack_2prong) {
        status_2prong = 0;
      }
      if (track.pt() < ptmintrack_3prong) {
        status_3prong = 0;
      }

      // eta cut
      if (status_2prong && abs(track.eta()) > etamax_2prong) {
        status_2prong = 0;
      }
      if (status_3prong && abs(track.eta()) > etamax_3prong) {
        status_3prong = 0;
      }

      // quality cut
      if (doCutQuality && (status_2prong || status_3prong)) {
        UChar_t clustermap = track.itsClusterMap();
        bool isselected = track.tpcNClsFound() >= d_tpcnclsfound &&
                          track.flags() & o2::aod::track::ITSrefit &&
                          (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1));
        if (!isselected) {
          status_2prong = 0;
          status_3prong = 0;
        }
      }

      // DCA cut
      array<float, 2> dca;
      if (status_2prong || status_3prong) {
        auto trackparvar0 = getTrackParCov(track);
        bool isprop = trackparvar0.propagateParamToDCA(vtxXYZ, d_bz, &dca, 100.); // get impact parameters
        if (!isprop) {
          status_2prong = 0;
          status_3prong = 0;
        }
        if (status_2prong && abs(dca[0]) < dcatoprimxymin_2prong) {
          status_2prong = 0;
        }
        if (status_3prong && abs(dca[0]) < dcatoprimxymin_3prong) {
          status_3prong = 0;
        }
      }

      // fill histograms
      if (b_dovalplots) {
        if (status_2prong == 1) {
          hpt_cuts_2prong->Fill(track.pt());
          hdcatoprimxy_cuts_2prong->Fill(dca[0]);
          heta_cuts_2prong->Fill(track.eta());
        }
        if (status_3prong == 1) {
          hpt_cuts_3prong->Fill(track.pt());
          hdcatoprimxy_cuts_3prong->Fill(dca[0]);
          heta_cuts_3prong->Fill(track.eta());
        }
      }

      // fill table row
      rowSelectedTrack(status_2prong, status_3prong, dca[0], dca[1]);
    }
  }
};

/// Pre-selection of 2-prong and 3-prong secondary vertices
struct HFTrackIndexSkimsCreator {
  Produces<aod::HfTrackIndexProng2> rowTrackIndexProng2;
  Produces<aod::HfTrackIndexProng3> rowTrackIndexProng3;

  Configurable<bool> b_dovalplots{"b_dovalplots", true, "fill histograms"};
  Configurable<int> do3prong{"do3prong", 0, "do 3 prong"};
  // event selection
  Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};
  // vertexing parameters
  Configurable<double> d_bz{"d_bz", 5., "magnetic field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations if chi2/chi2old > this"};
  // 2-prong cuts
  Configurable<double> cut2ProngPtCandMin{"cut2ProngPtCandMin", -1., "min. pT of the 2-prong candidate"};
  Configurable<double> cut2ProngInvMassD0Min{"cut2ProngInvMassD0Min", -1., "min. D0 candidate invariant mass"};
  Configurable<double> cut2ProngInvMassD0Max{"cut2ProngInvMassD0Max", -1., "max. D0 candidate invariant mass"};
  Configurable<double> cut2ProngCPAMin{"cut2ProngCPAMin", -2., "min. cosine of pointing angle"};
  Configurable<double> cut2ProngImpParProductMax{"cut2ProngImpParProductMax", 100., "max. product of imp. par. of D0 candidate prongs"};
  // 3-prong cuts
  Configurable<double> cut3ProngPtCandMin{"cut3ProngPtCandMin", -1., "min. pT of the 3-prong candidate"};
  Configurable<double> cut3ProngInvMassDPlusMin{"cut3ProngInvMassDPlusMin", 1.5, "min. D+ candidate invariant mass"};
  Configurable<double> cut3ProngInvMassDPlusMax{"cut3ProngInvMassDPlusMax", 2.1, "min. D+ candidate invariant mass"};
  Configurable<double> cut3ProngCPAMin{"cut3ProngCPAMin", -2., "min. cosine of pointing angle"};
  Configurable<double> cut3ProngDecLenMin{"cut3ProngDecLenMin", -1., "min. decay length"};

  // 2-prong histograms
  OutputObj<TH1F> hvtx2_x{TH1F("hvtx2_x", "2-prong candidates;#it{x}_{sec. vtx.} (cm);entries", 1000, -2., 2.)};
  OutputObj<TH1F> hvtx2_y{TH1F("hvtx2_y", "2-prong candidates;#it{y}_{sec. vtx.} (cm);entries", 1000, -2., 2.)};
  OutputObj<TH1F> hvtx2_z{TH1F("hvtx2_z", "2-prong candidates;#it{z}_{sec. vtx.} (cm);entries", 1000, -20., 20.)};
  OutputObj<TH1F> hmass2{TH1F("hmass2", "2-prong candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", 500, 0., 5.)};
  // 3-prong histograms
  OutputObj<TH1F> hvtx3_x{TH1F("hvtx3_x", "3-prong candidates;#it{x}_{sec. vtx.} (cm);entries", 1000, -2., 2.)};
  OutputObj<TH1F> hvtx3_y{TH1F("hvtx3_y", "3-prong candidates;#it{y}_{sec. vtx.} (cm);entries", 1000, -2., 2.)};
  OutputObj<TH1F> hvtx3_z{TH1F("hvtx3_z", "3-prong candidates;#it{z}_{sec. vtx.} (cm);entries", 1000, -20., 20.)};
  OutputObj<TH1F> hmass3{TH1F("hmass3", "3-prong candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", 500, 1.6, 2.1)};

  Filter filterSelectTracks = (aod::hf_seltrack::isSel2Prong == 1);
  using SelectedTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::HFSelTrack>>;
  // FIXME
  //Partition<SelectedTracks> tracksPos = aod::track::signed1Pt > 0.f;
  //Partition<SelectedTracks> tracksNeg = aod::track::signed1Pt < 0.f;

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double mass2PiK{0.};
  double mass2KPi{0.};
  double mass3PiKPi{0.};

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               SelectedTracks const& tracks)
  {
    int trigindex = int{triggerindex};
    if (trigindex != -1) {
      uint64_t triggerMask = collision.bc().triggerMask();
      bool isTriggerClassFired = triggerMask & 1ul << (trigindex - 1);
      if (!isTriggerClassFired) {
        return;
      }
    }

    // 2-prong vertex fitter
    o2::vertexing::DCAFitterN<2> df2;
    df2.setBz(d_bz);
    df2.setPropagateToPCA(b_propdca);
    df2.setMaxR(d_maxr);
    df2.setMaxDZIni(d_maxdzini);
    df2.setMinParamChange(d_minparamchange);
    df2.setMinRelChi2Change(d_minrelchi2change);
    df2.setUseAbsDCA(true);

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
      if (trackPos1.signed1Pt() < 0) {
        continue;
      }

      auto trackParVarPos1 = getTrackParCov(trackPos1);

      // first loop over negative tracks
      //for (auto trackNeg1 = tracksNeg.begin(); trackNeg1 != tracksNeg.end(); ++trackNeg1) {
      for (auto trackNeg1 = tracks.begin(); trackNeg1 != tracks.end(); ++trackNeg1) {
        if (trackNeg1.signed1Pt() > 0) {
          continue;
        }

        auto trackParVarNeg1 = getTrackParCov(trackNeg1);
        auto pVecCandProng2 = array{trackPos1.px() + trackNeg1.px(),
                                    trackPos1.py() + trackNeg1.py(),
                                    trackPos1.pz() + trackNeg1.pz()};
        bool isSelectedCand2Prong = true;

        // candidate pT cut
        if (RecoDecay::Pt(pVecCandProng2) < cut2ProngPtCandMin) {
          isSelectedCand2Prong = false;
        }

        if (isSelectedCand2Prong) {
          // reconstruct the 2-prong secondary vertex
          if (df2.process(trackParVarPos1, trackParVarNeg1) == 0) {
            continue;
          }

          // imp. par. product cut
          if (isSelectedCand2Prong && cut2ProngImpParProductMax < 100.) {
            if (trackPos1.dcaPrim0() * trackNeg1.dcaPrim0() > cut2ProngImpParProductMax) {
              isSelectedCand2Prong = false;
            }
          }

          // get secondary vertex
          const auto& secondaryVertex2 = df2.getPCACandidate();
          // get track momenta
          array<float, 3> pvec0;
          array<float, 3> pvec1;
          df2.getTrack(0).getPxPyPzGlo(pvec0);
          df2.getTrack(1).getPxPyPzGlo(pvec1);

          // CPA cut
          if (isSelectedCand2Prong && cut2ProngCPAMin > -2.) {
            pVecCandProng2 = RecoDecay::PVec(pvec0, pvec1);
            auto cpa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex2, pVecCandProng2);
            if (cpa < cut2ProngCPAMin) {
              isSelectedCand2Prong = false;
            }
          }

          if (isSelectedCand2Prong) {
            // calculate invariant masses
            auto arrMom = array{pvec0, pvec1};
            mass2PiK = RecoDecay::M(arrMom, array{massPi, massK});
            mass2KPi = RecoDecay::M(arrMom, array{massK, massPi});
          }

          // invariant-mass cut
          if (isSelectedCand2Prong && cut2ProngInvMassD0Min >= 0. && cut2ProngInvMassD0Max > 0.) {
            if ((mass2PiK < cut2ProngInvMassD0Min || mass2PiK > cut2ProngInvMassD0Max) &&
                (mass2KPi < cut2ProngInvMassD0Min || mass2KPi > cut2ProngInvMassD0Max)) {
              isSelectedCand2Prong = false;
            }
          }

          if (isSelectedCand2Prong) {
            // fill table row
            rowTrackIndexProng2(trackPos1.globalIndex(),
                                trackNeg1.globalIndex(), 1);

            // fill histograms
            if (b_dovalplots) {
              hvtx2_x->Fill(secondaryVertex2[0]);
              hvtx2_y->Fill(secondaryVertex2[1]);
              hvtx2_z->Fill(secondaryVertex2[2]);
              hmass2->Fill(mass2PiK);
              hmass2->Fill(mass2KPi);
            }
          }
        }

        // 3-prong vertex reconstruction
        if (do3prong == 1) {
          if (trackPos1.isSel3Prong() == 0) {
            continue;
          }
          if (trackNeg1.isSel3Prong() == 0) {
            continue;
          }

          // second loop over positive tracks
          //for (auto trackPos2 = trackPos1 + 1; trackPos2 != tracksPos.end(); ++trackPos2) {
          for (auto trackPos2 = trackPos1 + 1; trackPos2 != tracks.end(); ++trackPos2) {
            if (trackPos2.signed1Pt() < 0) {
              continue;
            }
            if (trackPos2.isSel3Prong() == 0) {
              continue;
            }

            auto pVecCandProng3Pos = RecoDecay::PVec(pVecCandProng2, array{trackPos2.px(), trackPos2.py(), trackPos2.pz()});

            // candidate pT cut
            if (RecoDecay::Pt(pVecCandProng3Pos) < cut3ProngPtCandMin) {
              continue;
            }

            // invariant-mass cut
            if (cut3ProngInvMassDPlusMin >= 0. && cut3ProngInvMassDPlusMax > 0.) {
              // calculate invariant mass
              auto arr3Mom = array{
                array{trackPos1.px(), trackPos1.py(), trackPos1.pz()},
                array{trackNeg1.px(), trackNeg1.py(), trackNeg1.pz()},
                array{trackPos2.px(), trackPos2.py(), trackPos2.pz()}};
              mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});
              if (mass3PiKPi < cut3ProngInvMassDPlusMin || mass3PiKPi > cut3ProngInvMassDPlusMax) {
                continue;
              }
            }

            // reconstruct the 3-prong secondary vertex
            auto trackParVarPos2 = getTrackParCov(trackPos2);
            if (df3.process(trackParVarPos1, trackParVarNeg1, trackParVarPos2) == 0) {
              continue;
            }

            // get secondary vertex
            const auto& secondaryVertex3 = df3.getPCACandidate();
            // get track momenta
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);

            // CPA cut
            if (cut3ProngCPAMin > -2.) {
              pVecCandProng3Pos = RecoDecay::PVec(pvec0, pvec1, pvec2);
              auto cpa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3, pVecCandProng3Pos);
              if (cpa < cut3ProngCPAMin) {
                continue;
              }
            }

            // decay length cut
            if (cut3ProngDecLenMin > 0.) {
              auto decayLength = RecoDecay::distance(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3);
              if (decayLength < cut3ProngDecLenMin) {
                continue;
              }
            }

            // fill table row
            rowTrackIndexProng3(trackPos1.globalIndex(),
                                trackNeg1.globalIndex(),
                                trackPos2.globalIndex(), 2);

            // fill histograms
            if (b_dovalplots) {
              hvtx3_x->Fill(secondaryVertex3[0]);
              hvtx3_y->Fill(secondaryVertex3[1]);
              hvtx3_z->Fill(secondaryVertex3[2]);

              // calculate invariant mass
              hmass3->Fill(RecoDecay::M(array{pvec0, pvec1, pvec2}, array{massPi, massK, massPi}));
            }
          }

          // second loop over negative tracks
          //for (auto trackNeg2 = trackNeg1 + 1; trackNeg2 != tracksNeg.end(); ++trackNeg2) {
          for (auto trackNeg2 = trackNeg1 + 1; trackNeg2 != tracks.end(); ++trackNeg2) {
            if (trackNeg2.signed1Pt() > 0) {
              continue;
            }
            if (trackNeg2.isSel3Prong() == 0) {
              continue;
            }

            auto pVecCandProng3Neg = RecoDecay::PVec(pVecCandProng2, array{trackNeg2.px(), trackNeg2.py(), trackNeg2.pz()});

            // candidate pT cut
            if (RecoDecay::Pt(pVecCandProng3Neg) < cut3ProngPtCandMin) {
              continue;
            }

            // invariant-mass cut
            if (cut3ProngInvMassDPlusMin >= 0. && cut3ProngInvMassDPlusMax > 0.) {
              // calculate invariant mass
              auto arr3Mom = array{
                array{trackNeg1.px(), trackNeg1.py(), trackNeg1.pz()},
                array{trackPos1.px(), trackPos1.py(), trackPos1.pz()},
                array{trackNeg2.px(), trackNeg2.py(), trackNeg2.pz()}};
              mass3PiKPi = RecoDecay::M(std::move(arr3Mom), array{massPi, massK, massPi});
              if (mass3PiKPi < cut3ProngInvMassDPlusMin || mass3PiKPi > cut3ProngInvMassDPlusMax) {
                continue;
              }
            }

            // reconstruct the 3-prong secondary vertex
            auto trackParVarNeg2 = getTrackParCov(trackNeg2);
            if (df3.process(trackParVarNeg1, trackParVarPos1, trackParVarNeg2) == 0) {
              continue;
            }

            // get secondary vertex
            const auto& secondaryVertex3 = df3.getPCACandidate();
            // get track momenta
            array<float, 3> pvec0;
            array<float, 3> pvec1;
            array<float, 3> pvec2;
            df3.getTrack(0).getPxPyPzGlo(pvec0);
            df3.getTrack(1).getPxPyPzGlo(pvec1);
            df3.getTrack(2).getPxPyPzGlo(pvec2);

            // CPA cut
            if (cut3ProngCPAMin > -2.) {
              pVecCandProng3Neg = RecoDecay::PVec(pvec0, pvec1, pvec2);
              auto cpa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3, pVecCandProng3Neg);
              if (cpa < cut3ProngCPAMin) {
                continue;
              }
            }

            // decay length cut
            if (cut3ProngDecLenMin > 0.) {
              auto decayLength = RecoDecay::distance(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex3);
              if (decayLength < cut3ProngDecLenMin) {
                continue;
              }
            }

            // fill table row
            rowTrackIndexProng3(trackNeg1.globalIndex(),
                                trackPos1.globalIndex(),
                                trackNeg2.globalIndex(), 2);

            // fill histograms
            if (b_dovalplots) {
              hvtx3_x->Fill(secondaryVertex3[0]);
              hvtx3_y->Fill(secondaryVertex3[1]);
              hvtx3_z->Fill(secondaryVertex3[2]);

              // calculate invariant mass
              hmass3->Fill(RecoDecay::M(array{pvec0, pvec1, pvec2}, array{massPi, massK, massPi}));
            }
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
