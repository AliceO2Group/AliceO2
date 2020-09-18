// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// This task re-reconstructs the V0s and cascades

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Analysis/SecondaryVertexHF.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Analysis/RecoDecay.h"
#include "Analysis/trackUtilities.h"
#include "PID/PIDResponse.h"
#include "Analysis/StrangenessTables.h"
#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Centrality.h"

#include <TFile.h>
#include <TLorentzVector.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
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

struct lambdakzerofinder {
  Produces<aod::V0Data> v0data;
  Produces<aod::V0FinderData> v0finderdata;

  OutputObj<TH1F> hCandPerEvent{TH1F("hCandPerEvent", "", 1000, 0, 1000)};

  OutputObj<TH1F> hV0PosCrossedRows{TH1F("hV0PosCrossedRows", "", 160, 0, 160)};
  OutputObj<TH1F> hV0NegCrossedRows{TH1F("hV0NegCrossedRows", "", 160, 0, 160)};
  OutputObj<TH1F> hV0CosPA{TH1F("hV0CosPA", "", 2000, 0.9, 1)};
  OutputObj<TH1F> hV0Radius{TH1F("hV0Radius", "", 2000, 0, 200)};
  OutputObj<TH1F> hV0DCADaughters{TH1F("hV0DCADaughters", "", 200, 0, 2)};
  OutputObj<TH1F> hV0PosDCAxy{TH1F("hV0PosDCAxy", "", 200, 0, 5)};
  OutputObj<TH1F> hV0NegDCAxy{TH1F("hV0NegDCAxy", "", 200, 0, 5)};

  //Configurables
  Configurable<double> d_bz{"d_bz", +5.0, "bz field"};
  Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};

  //Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.995, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};

  //using myTracks = soa::Filtered<aod::Tracks>;
  //using myTracks = soa::Filtered<aod::fullTracks>;

  Partition<aod::FullTracks> goodPosTracks = aod::track::signed1Pt > 0.0f;
  Partition<aod::FullTracks> goodNegTracks = aod::track::signed1Pt < 0.0f;

  /// Extracts dca in the XY plane
  /// \return dcaXY
  template <typename T, typename U>
  auto getdcaXY(const T& track, const U& coll)
  {
    //Calculate DCAs
    auto sinAlpha = sin(track.alpha());
    auto cosAlpha = cos(track.alpha());
    auto globalX = track.x() * cosAlpha - track.y() * sinAlpha;
    auto globalY = track.x() * sinAlpha + track.y() * cosAlpha;
    return sqrt(pow((globalX - coll[0]), 2) +
                pow((globalY - coll[1]), 2));
  }

  void process(aod::Collision const& collision,
               aod::FullTracks const& tracks)
  {
    //Define o2 fitter, 2-prong
    o2::vertexing::DCAFitterN<2> fitter;
    fitter.setBz(d_bz);
    fitter.setPropagateToPCA(true);
    fitter.setMaxR(200.);
    fitter.setMinParamChange(1e-3);
    fitter.setMinRelChi2Change(0.9);
    fitter.setMaxDZIni(1e9);
    fitter.setMaxChi2(1e9);
    fitter.setUseAbsDCA(d_UseAbsDCA);

    Long_t lNCand = 0;

    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};

    for (auto& t0 : goodPosTracks) {
      auto thisdcapostopv = getdcaXY(t0, pVtx);
      if (thisdcapostopv < dcapostopv)
        continue;
      if (t0.tpcNClsCrossedRows() < 70)
        continue;
      for (auto& t1 : goodPosTracks) {
        auto thisdcanegtopv = getdcaXY(t1, pVtx);
        if (thisdcanegtopv < dcanegtopv)
          continue;
        if (t1.tpcNClsCrossedRows() < 70)
          continue;

        auto Track1 = getTrackParCov(t0);
        auto Track2 = getTrackParCov(t1);

        //Try to progate to dca
        int nCand = fitter.process(Track1, Track2);
        if (nCand == 0)
          continue;
        const auto& vtx = fitter.getPCACandidate();

        //Fiducial: min radius
        auto thisv0radius = TMath::Sqrt(TMath::Power(vtx[0], 2) + TMath::Power(vtx[1], 2));
        if (thisv0radius < v0radius)
          continue;

        //DCA V0 daughters
        auto thisdcav0dau = fitter.getChi2AtPCACandidate();
        if (thisdcav0dau > dcav0dau)
          continue;

        std::array<float, 3> pos = {0.};
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        for (int i = 0; i < 3; i++)
          pos[i] = vtx[i];
        fitter.getTrack(0).getPxPyPzGlo(pvec0);
        fitter.getTrack(1).getPxPyPzGlo(pvec1);

        auto thisv0cospa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()},
                                          array{vtx[0], vtx[1], vtx[2]}, array{pvec0[0] + pvec1[0], pvec0[1] + pvec1[1], pvec0[2] + pvec1[2]});
        if (thisv0cospa < v0cospa)
          continue;

        hV0PosCrossedRows->Fill(t0.tpcNClsCrossedRows());
        hV0NegCrossedRows->Fill(t1.tpcNClsCrossedRows());
        hV0PosDCAxy->Fill(thisdcapostopv);
        hV0NegDCAxy->Fill(thisdcanegtopv);
        hV0CosPA->Fill(thisv0cospa);
        hV0Radius->Fill(thisv0radius);
        hV0DCADaughters->Fill(thisdcav0dau);

        lNCand++;
        v0finderdata(t0.collisionId());
        v0data(pos[0], pos[1], pos[2],
               pvec0[0], pvec0[1], pvec0[2],
               pvec1[0], pvec1[1], pvec1[2],
               fitter.getChi2AtPCACandidate(),
               getdcaXY(t0, pVtx),
               getdcaXY(t1, pVtx));
      }
    }
    hCandPerEvent->Fill(lNCand);
  }
};

struct lambdakzerofinderQA {
  //Basic checks
  //Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.998, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", .6, "DCA V0 Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};

  OutputObj<TH1F> hCandPerEvent{TH1F("hCandPerEvent", "", 1000, 0, 1000)};

  OutputObj<TH1F> hV0Radius{TH1F("hV0Radius", "", 1000, 0.0, 100)};
  OutputObj<TH1F> hV0CosPA{TH1F("hV0CosPA", "", 1000, 0.95, 1.0)};
  OutputObj<TH1F> hDCAPosToPV{TH1F("hDCAPosToPV", "", 1000, 0.0, 10.0)};
  OutputObj<TH1F> hDCANegToPV{TH1F("hDCANegToPV", "", 1000, 0.0, 10.0)};
  OutputObj<TH1F> hDCAV0Dau{TH1F("hDCAV0Dau", "", 1000, 0.0, 10.0)};

  OutputObj<TH3F> h3dMassK0Short{TH3F("h3dMassK0Short", "", 20, 0, 100, 200, 0, 10, 200, 0.450, 0.550)};
  OutputObj<TH3F> h3dMassLambda{TH3F("h3dMassLambda", "", 20, 0, 100, 200, 0, 10, 200, 1.115 - 0.100, 1.115 + 0.100)};
  OutputObj<TH3F> h3dMassAntiLambda{TH3F("h3dMassAntiLambda", "", 20, 0, 100, 200, 0, 10, 200, 1.115 - 0.100, 1.115 + 0.100)};

  Filter preFilterV0 = aod::v0data::dcapostopv > dcapostopv&&
                                                   aod::v0data::dcanegtopv > dcanegtopv&& aod::v0data::dcaV0daughters < dcav0dau;

  ///Connect to V0FinderData: newly indexed, note: V0DataExt table incompatible with standard V0 table!
  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator const& collision,
               soa::Filtered<soa::Join<aod::V0FinderData, aod::V0DataExt>> const& fullV0s)
  {
    if (!collision.alias()[kINT7])
      return;
    if (!collision.sel7())
      return;

    Long_t lNCand = 0;
    for (auto& v0 : fullV0s) {
      if (v0.v0radius() > v0radius && v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) > v0cospa) {
        hV0Radius->Fill(v0.v0radius());
        hV0CosPA->Fill(v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()));
        hDCAPosToPV->Fill(v0.dcapostopv());
        hDCANegToPV->Fill(v0.dcanegtopv());
        hDCAV0Dau->Fill(v0.dcaV0daughters());

        if (TMath::Abs(v0.yLambda()) < 0.5) {
          h3dMassLambda->Fill(collision.centV0M(), v0.pt(), v0.mLambda());
          h3dMassAntiLambda->Fill(collision.centV0M(), v0.pt(), v0.mAntiLambda());
        }
        if (TMath::Abs(v0.yK0Short()) < 0.5)
          h3dMassK0Short->Fill(collision.centV0M(), v0.pt(), v0.mK0Short());
        lNCand++;
      }
    }
    hCandPerEvent->Fill(lNCand);
  }
};

/// Extends the v0data table with expression columns
struct lambdakzeroinitializer {
  Spawns<aod::V0DataExt> v0dataext;
  void init(InitContext const&) {}
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<lambdakzerofinder>("lf-lambdakzerofinder"),
    adaptAnalysisTask<lambdakzerofinderQA>("lf-lambdakzerofinderQA"),
    adaptAnalysisTask<lambdakzeroinitializer>("lf-lambdakzeroinitializer")};
}
