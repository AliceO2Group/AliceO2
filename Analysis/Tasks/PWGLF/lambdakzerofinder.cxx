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
// V0 Finder task
// ==============
//
// This code loops over positive and negative tracks and finds
// valid V0 candidates from scratch using a certain set of
// minimum (configurable) selection criteria.
//
// It is different than the producer: the producer merely
// loops over an *existing* list of V0s (pos+neg track
// indices) and calculates the corresponding full V0 information
//
// In both cases, any analysis should loop over the "V0Data"
// table as that table contains all information.
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "AnalysisCore/RecoDecay.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/StrangenessTables.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"

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

namespace o2::aod
{
namespace v0goodpostracks
{
DECLARE_SOA_INDEX_COLUMN_FULL(GoodTrack, goodTrack, int, FullTracks, "fGoodTrackID");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(DCAXY, dcaXY, float);
} // namespace v0goodpostracks
DECLARE_SOA_TABLE(V0GoodPosTracks, "AOD", "V0GOODPOSTRACKS", o2::soa::Index<>, v0goodpostracks::GoodTrackId, v0goodpostracks::CollisionId, v0goodpostracks::DCAXY);
namespace v0goodnegtracks
{
DECLARE_SOA_INDEX_COLUMN_FULL(GoodTrack, goodTrack, int, FullTracks, "fGoodTrackID");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(DCAXY, dcaXY, float);
} // namespace v0goodnegtracks
DECLARE_SOA_TABLE(V0GoodNegTracks, "AOD", "V0GOODNEGTRACKS", o2::soa::Index<>, v0goodnegtracks::GoodTrackId, v0goodnegtracks::CollisionId, v0goodnegtracks::DCAXY);
} // namespace o2::aod

struct lambdakzeroprefilter {
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<int> tpcrefit{"tpcrefit", 1, "demand TPC refit"};

  Produces<aod::V0GoodPosTracks> v0GoodPosTracks;
  Produces<aod::V0GoodNegTracks> v0GoodNegTracks;

  //still exhibiting issues? To be checked
  //Partition<soa::Join<aod::FullTracks, aod::TracksExtended>> goodPosTracks = aod::track::signed1Pt > 0.0f && aod::track::dcaXY > dcapostopv;
  //Partition<soa::Join<aod::FullTracks, aod::TracksExtended>> goodNegTracks = aod::track::signed1Pt < 0.0f && aod::track::dcaXY < -dcanegtopv;

  void process(aod::Collision const& collision,
               soa::Join<aod::FullTracks, aod::TracksExtended> const& tracks)
  {
    for (auto& t0 : tracks) {
      if (tpcrefit) {
        if (!(t0.trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
      }
      if (t0.tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      if (t0.signed1Pt() > 0.0f) {
        if (fabs(t0.dcaXY()) < dcapostopv) {
          continue;
        }
        v0GoodPosTracks(t0.globalIndex(), t0.collisionId(), t0.dcaXY());
      }
      if (t0.signed1Pt() < 0.0f) {
        if (fabs(t0.dcaXY()) < dcanegtopv) {
          continue;
        }
        v0GoodNegTracks(t0.globalIndex(), t0.collisionId(), -t0.dcaXY());
      }
    }
  }
};

struct lambdakzerofinder {
  Produces<aod::V0Data> v0data;

  OutputObj<TH1F> hCandPerEvent{TH1F("hCandPerEvent", "", 1000, 0, 1000)};

  //Configurables
  Configurable<double> d_bz{"d_bz", +5.0, "bz field"};
  Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};

  //Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.995, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
  Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};

  void process(aod::Collision const& collision, aod::FullTracks const& tracks,
               aod::V0GoodPosTracks const& ptracks, aod::V0GoodNegTracks const& ntracks)
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

    for (auto& t0id : ptracks) { //FIXME: turn into combination(...)
      auto t0 = t0id.goodTrack();
      auto Track1 = getTrackParCov(t0);
      for (auto& t1id : ntracks) {
        auto t1 = t1id.goodTrack();
        auto Track2 = getTrackParCov(t1);

        //Try to progate to dca
        int nCand = fitter.process(Track1, Track2);
        if (nCand == 0) {
          continue;
        }
        const auto& vtx = fitter.getPCACandidate();

        //Fiducial: min radius
        auto thisv0radius = TMath::Sqrt(TMath::Power(vtx[0], 2) + TMath::Power(vtx[1], 2));
        if (thisv0radius < v0radius) {
          continue;
        }

        //DCA V0 daughters
        auto thisdcav0dau = fitter.getChi2AtPCACandidate();
        if (thisdcav0dau > dcav0dau) {
          continue;
        }

        std::array<float, 3> pos = {0.};
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        for (int i = 0; i < 3; i++) {
          pos[i] = vtx[i];
        }
        fitter.getTrack(0).getPxPyPzGlo(pvec0);
        fitter.getTrack(1).getPxPyPzGlo(pvec1);

        auto thisv0cospa = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()},
                                          array{vtx[0], vtx[1], vtx[2]}, array{pvec0[0] + pvec1[0], pvec0[1] + pvec1[1], pvec0[2] + pvec1[2]});
        if (thisv0cospa < v0cospa) {
          continue;
        }

        lNCand++;
        v0data(t0.globalIndex(), t1.globalIndex(), t0.collisionId(),
               fitter.getTrack(0).getX(), fitter.getTrack(1).getX(),
               pos[0], pos[1], pos[2],
               pvec0[0], pvec0[1], pvec0[2],
               pvec1[0], pvec1[1], pvec1[2],
               fitter.getChi2AtPCACandidate(),
               t0id.dcaXY(), t1id.dcaXY());
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

  //FIXME: figure out why this does not work?
  //Filter preFilter1 = aod::v0data::dcapostopv > dcapostopv;
  //Filter preFilter2 = aod::v0data::dcanegtopv > dcanegtopv;
  //Filter preFilter3 = aod::v0data::dcaV0daughters < dcav0dau;

  ///Connect to V0Data: newly indexed, note: V0DataExt table incompatible with standard V0 table!
  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator const& collision,
               aod::V0DataExt const& fullV0s)
  {
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    Long_t lNCand = 0;
    for (auto& v0 : fullV0s) {
      if (v0.v0radius() > v0radius && v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) > v0cospa && v0.dcapostopv() > dcapostopv && v0.dcanegtopv() > dcanegtopv && v0.dcaV0daughters() > dcav0dau) {
        hV0Radius->Fill(v0.v0radius());
        hV0CosPA->Fill(v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()));
        hDCAPosToPV->Fill(v0.dcapostopv());
        hDCANegToPV->Fill(v0.dcanegtopv());
        hDCAV0Dau->Fill(v0.dcaV0daughters());

        if (TMath::Abs(v0.yLambda()) < 0.5) {
          h3dMassLambda->Fill(collision.centV0M(), v0.pt(), v0.mLambda());
          h3dMassAntiLambda->Fill(collision.centV0M(), v0.pt(), v0.mAntiLambda());
        }
        if (TMath::Abs(v0.yK0Short()) < 0.5) {
          h3dMassK0Short->Fill(collision.centV0M(), v0.pt(), v0.mK0Short());
        }
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
    adaptAnalysisTask<lambdakzeroprefilter>("lf-lambdakzeroprefilter"),
    adaptAnalysisTask<lambdakzerofinder>("lf-lambdakzerofinder"),
    adaptAnalysisTask<lambdakzerofinderQA>("lf-lambdakzerofinderQA"),
    adaptAnalysisTask<lambdakzeroinitializer>("lf-lambdakzeroinitializer")};
}
