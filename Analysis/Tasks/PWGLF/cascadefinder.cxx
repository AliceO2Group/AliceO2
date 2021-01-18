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
// Cascade Finder task
// ===================
//
// This code loops over existing V0s and bachelor tracks and finds
// valid cascade candidates from scratch using a certain set of
// minimum (configurable) selection criteria.
//
// It is different than the producer: the producer merely
// loops over an *existing* list of cascades (V0+bachelor track
// indices) and calculates the corresponding full cascade information
//
// In both cases, any analysis should loop over the "CascData"
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
#include "AnalysisDataModel/HFSecondaryVertex.h"
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

namespace cascgoodpostracks
{
DECLARE_SOA_INDEX_COLUMN_FULL(GoodPosTrack, goodPosTrack, int, FullTracks, "fGoodPosTrackID");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(DCAXY, dcaXY, float);
} // namespace cascgoodpostracks
DECLARE_SOA_TABLE(CascGoodPosTracks, "AOD", "CASCGOODPTRACKS", o2::soa::Index<>, cascgoodpostracks::GoodPosTrackId, cascgoodpostracks::CollisionId, cascgoodpostracks::DCAXY);
namespace cascgoodnegtracks
{
DECLARE_SOA_INDEX_COLUMN_FULL(GoodNegTrack, goodNegTrack, int, FullTracks, "fGoodNegTrackID");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(DCAXY, dcaXY, float);
} // namespace cascgoodnegtracks
DECLARE_SOA_TABLE(CascGoodNegTracks, "AOD", "CASCGOODNTRACKS", o2::soa::Index<>, cascgoodnegtracks::GoodNegTrackId, cascgoodnegtracks::CollisionId, cascgoodnegtracks::DCAXY);
namespace cascgoodlambdas
{
DECLARE_SOA_INDEX_COLUMN_FULL(GoodLambda, goodLambda, int, V0DataExt, "fGoodLambdaId");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
} // namespace cascgoodlambdas
DECLARE_SOA_TABLE(CascGoodLambdas, "AOD", "CASCGOODLAM", o2::soa::Index<>, cascgoodlambdas::GoodLambdaId, cascgoodlambdas::CollisionId);
namespace cascgoodantilambdas
{
DECLARE_SOA_INDEX_COLUMN_FULL(GoodAntiLambda, goodAntiLambda, int, V0DataExt, "fGoodAntiLambdaId");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
} // namespace cascgoodantilambdas
DECLARE_SOA_TABLE(CascGoodAntiLambdas, "AOD", "CASCGOODALAM", o2::soa::Index<>, cascgoodantilambdas::GoodAntiLambdaId, cascgoodantilambdas::CollisionId);
} // namespace o2::aod

struct cascadeprefilter {
  Configurable<float> dcabachtopv{"dcabachtopv", .1, "DCA Bach To PV"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<float> dcav0topv{"dcav0topv", .1, "DCA V0 To PV"};
  Configurable<double> cospaV0{"cospaV0", .98, "CosPA V0"};
  Configurable<double> v0radius{"v0radius", 0.9, "v0radius"};
  Configurable<float> lambdamasswindow{"lambdamasswindow", .006, "Distance from Lambda mass"};
  Configurable<float> dcav0dau{"dcav0dau", .6, "DCA V0 Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};

  Produces<aod::CascGoodLambdas> cascGoodLambdas;
  Produces<aod::CascGoodAntiLambdas> cascGoodAntiLambdas;
  Produces<aod::CascGoodPosTracks> cascGoodPosTracks;
  Produces<aod::CascGoodNegTracks> cascGoodNegTracks;

  Partition<soa::Join<aod::FullTracks, aod::TracksExtended>> goodPosTracks = aod::track::signed1Pt > 0.0f && aod::track::dcaXY > dcabachtopv;
  Partition<soa::Join<aod::FullTracks, aod::TracksExtended>> goodNegTracks = aod::track::signed1Pt < 0.0f && aod::track::dcaXY < -dcabachtopv;

  Partition<aod::V0DataExt> goodV0s = aod::v0data::dcapostopv > dcapostopv&& aod::v0data::dcanegtopv > dcanegtopv&& aod::v0data::dcaV0daughters < dcav0dau;

  using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtended>;

  void process(aod::Collision const& collision,
               FullTracksExt const& tracks,
               aod::V0DataExt const& V0s)
  {
    for (auto& t0 : goodPosTracks) {
      if (!(t0.trackType() & o2::aod::track::TPCrefit)) {
        continue; //TPC refit
      }
      if (t0.tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      cascGoodPosTracks(t0.globalIndex(), t0.collisionId(), t0.dcaXY());
    }
    for (auto& t0 : goodNegTracks) {
      if (!(t0.trackType() & o2::aod::track::TPCrefit)) {
        continue; //TPC refit
      }
      if (t0.tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      cascGoodNegTracks(t0.globalIndex(), t0.collisionId(), -t0.dcaXY());
    }
    for (auto& v0 : goodV0s) {
      if (v0.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) < cospaV0) {
        continue;
      }
      if (v0.dcav0topv(collision.posX(), collision.posY(), collision.posZ()) < dcav0topv) {
        continue;
      }
      if (v0.dcaV0daughters() > dcav0dau) {
        continue;
      }
      if (v0.v0radius() < v0radius) {
        continue;
      }

      if (fabs(v0.mLambda() - 1.116) < lambdamasswindow) {
        cascGoodLambdas(v0.globalIndex(), v0.collisionId());
      }
      if (fabs(v0.mAntiLambda() - 1.116) < lambdamasswindow) {
        cascGoodAntiLambdas(v0.globalIndex(), v0.collisionId());
      }
    }
  }
};

struct cascadefinder {
  Produces<aod::CascData> cascdata;

  OutputObj<TH1F> hCandPerEvent{TH1F("hCandPerEvent", "", 100, 0, 100)};

  //Configurables
  Configurable<double> d_bz{"d_bz", +5.0, "bz field"};
  Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};

  //Selection criteria
  Configurable<double> v0cospa{"casccospa", 0.998, "Casc CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcacascdau", 1.0, "DCA Casc Daughters"};
  Configurable<float> v0radius{"cascradius", 1.0, "cascradius"};

  //Process: subscribes to a lot of things!
  void process(aod::Collision const& collision,
               aod::FullTracks const& tracks,
               aod::V0DataExt const& V0s,
               aod::CascGoodLambdas const& lambdas,
               aod::CascGoodAntiLambdas const& antiLambdas,
               aod::CascGoodPosTracks const& pBachtracks,
               aod::CascGoodNegTracks const& nBachtracks)
  {
    //Define o2 fitter, 2-prong
    o2::vertexing::DCAFitterN<2> fitterV0, fitterCasc;
    fitterV0.setBz(d_bz);
    fitterV0.setPropagateToPCA(true);
    fitterV0.setMaxR(200.);
    fitterV0.setMinParamChange(1e-3);
    fitterV0.setMinRelChi2Change(0.9);
    fitterV0.setMaxDZIni(1e9);
    fitterV0.setMaxChi2(1e9);
    fitterV0.setUseAbsDCA(d_UseAbsDCA);

    fitterCasc.setBz(d_bz);
    fitterCasc.setPropagateToPCA(true);
    fitterCasc.setMaxR(200.);
    fitterCasc.setMinParamChange(1e-3);
    fitterCasc.setMinRelChi2Change(0.9);
    fitterCasc.setMaxDZIni(1e9);
    fitterCasc.setMaxChi2(1e9);
    fitterCasc.setUseAbsDCA(d_UseAbsDCA);

    Long_t lNCand = 0;

    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};
    std::array<float, 3> pos = {0.};
    std::array<float, 3> posXi = {0.};
    std::array<float, 3> pvecpos = {0.};
    std::array<float, 3> pvecneg = {0.};
    std::array<float, 3> pvecbach = {0.};

    //Cascades first
    for (auto& v0id : lambdas) {
      //required: de-reference the tracks for cascade building
      auto v0 = v0id.goodLambda();
      auto pTrack = getTrackParCov(v0.posTrack());
      auto nTrack = getTrackParCov(v0.negTrack());
      //Let's do the slow part first: the V0 recalculation from scratch
      int nCand = fitterV0.process(pTrack, nTrack);
      if (nCand != 0) {
        fitterV0.propagateTracksToVertex();
        const auto& v0vtx = fitterV0.getPCACandidate();
        for (int i = 0; i < 3; i++) {
          pos[i] = v0vtx[i];
        }

        std::array<float, 21> cov0 = {0};
        std::array<float, 21> cov1 = {0};
        std::array<float, 21> covV0 = {0};

        //Covariance matrix calculation
        const int momInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
        fitterV0.getTrack(0).getPxPyPzGlo(pvecpos);
        fitterV0.getTrack(1).getPxPyPzGlo(pvecneg);
        fitterV0.getTrack(0).getCovXYZPxPyPzGlo(cov0);
        fitterV0.getTrack(1).getCovXYZPxPyPzGlo(cov1);
        for (int i = 0; i < 6; i++) {
          int j = momInd[i];
          covV0[j] = cov0[j] + cov1[j];
        }
        auto covVtxV0 = fitterV0.calcPCACovMatrix();
        covV0[0] = covVtxV0(0, 0);
        covV0[1] = covVtxV0(1, 0);
        covV0[2] = covVtxV0(1, 1);
        covV0[3] = covVtxV0(2, 0);
        covV0[4] = covVtxV0(2, 1);
        covV0[5] = covVtxV0(2, 2);

        const std::array<float, 3> vertex = {(float)v0vtx[0], (float)v0vtx[1], (float)v0vtx[2]};
        const std::array<float, 3> momentum = {pvecpos[0] + pvecneg[0], pvecpos[1] + pvecneg[1], pvecpos[2] + pvecneg[2]};

        auto tV0 = o2::track::TrackParCov(vertex, momentum, covV0, 0);
        tV0.setQ2Pt(0); //No bending, please

        for (auto& t0id : nBachtracks) {
          auto t0 = t0id.goodNegTrack();
          auto bTrack = getTrackParCov(t0);

          int nCand2 = fitterCasc.process(tV0, bTrack);
          if (nCand2 != 0) {
            fitterCasc.propagateTracksToVertex();
            const auto& cascvtx = fitterCasc.getPCACandidate();
            for (int i = 0; i < 3; i++) {
              posXi[i] = cascvtx[i];
            }
            fitterCasc.getTrack(1).getPxPyPzGlo(pvecbach);

            lNCand++;
            //If we got here, it means this is a good candidate!
            cascdata(v0.globalIndex(), v0.posTrack().globalIndex(), v0.negTrack().collisionId(),
                     -1, posXi[0], posXi[1], posXi[2], pos[0], pos[1], pos[2],
                     pvecpos[0], pvecpos[1], pvecpos[2],
                     pvecneg[0], pvecneg[1], pvecneg[2],
                     pvecbach[0], pvecbach[1], pvecbach[2],
                     fitterV0.getChi2AtPCACandidate(), fitterCasc.getChi2AtPCACandidate(),
                     v0.dcapostopv(),
                     v0.dcanegtopv(),
                     t0id.dcaXY());
          } //end if cascade recoed
        }   //end loop over bachelor
      }     //end if v0 recoed
    }       //end loop over cascades

    //Anticascades
    for (auto& v0id : antiLambdas) {
      //required: de-reference the tracks for cascade building
      auto v0 = v0id.goodAntiLambda();
      auto pTrack = getTrackParCov(v0.posTrack());
      auto nTrack = getTrackParCov(v0.negTrack());
      //Let's do the slow part first: the V0 recalculation from scratch
      int nCand = fitterV0.process(pTrack, nTrack);
      if (nCand != 0) {
        fitterV0.propagateTracksToVertex();
        const auto& v0vtx = fitterV0.getPCACandidate();
        for (int i = 0; i < 3; i++) {
          pos[i] = v0vtx[i];
        }

        std::array<float, 21> cov0 = {0};
        std::array<float, 21> cov1 = {0};
        std::array<float, 21> covV0 = {0};

        //Covariance matrix calculation
        const int momInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
        fitterV0.getTrack(0).getPxPyPzGlo(pvecpos);
        fitterV0.getTrack(1).getPxPyPzGlo(pvecneg);
        fitterV0.getTrack(0).getCovXYZPxPyPzGlo(cov0);
        fitterV0.getTrack(1).getCovXYZPxPyPzGlo(cov1);
        for (int i = 0; i < 6; i++) {
          int j = momInd[i];
          covV0[j] = cov0[j] + cov1[j];
        }
        auto covVtxV0 = fitterV0.calcPCACovMatrix();
        covV0[0] = covVtxV0(0, 0);
        covV0[1] = covVtxV0(1, 0);
        covV0[2] = covVtxV0(1, 1);
        covV0[3] = covVtxV0(2, 0);
        covV0[4] = covVtxV0(2, 1);
        covV0[5] = covVtxV0(2, 2);

        const std::array<float, 3> vertex = {(float)v0vtx[0], (float)v0vtx[1], (float)v0vtx[2]};
        const std::array<float, 3> momentum = {pvecpos[0] + pvecneg[0], pvecpos[1] + pvecneg[1], pvecpos[2] + pvecneg[2]};

        auto tV0 = o2::track::TrackParCov(vertex, momentum, covV0, 0);
        tV0.setQ2Pt(0); //No bending, please

        for (auto& t0id : pBachtracks) {
          auto t0 = t0id.goodPosTrack();
          auto bTrack = getTrackParCov(t0);

          int nCand2 = fitterCasc.process(tV0, bTrack);
          if (nCand2 != 0) {
            fitterCasc.propagateTracksToVertex();
            const auto& cascvtx = fitterCasc.getPCACandidate();
            for (int i = 0; i < 3; i++) {
              posXi[i] = cascvtx[i];
            }
            fitterCasc.getTrack(1).getPxPyPzGlo(pvecbach);

            lNCand++;
            //If we got here, it means this is a good candidate!
            cascdata(v0.globalIndex(), v0.posTrack().globalIndex(), v0.negTrack().collisionId(),
                     +1, posXi[0], posXi[1], posXi[2], pos[0], pos[1], pos[2],
                     pvecpos[0], pvecpos[1], pvecpos[2],
                     pvecneg[0], pvecneg[1], pvecneg[2],
                     pvecbach[0], pvecbach[1], pvecbach[2],
                     fitterV0.getChi2AtPCACandidate(), fitterCasc.getChi2AtPCACandidate(),
                     v0.dcapostopv(),
                     v0.dcanegtopv(),
                     t0id.dcaXY());
          } //end if cascade recoed
        }   //end loop over bachelor
      }     //end if v0 recoed
    }       //end loop over anticascades

    hCandPerEvent->Fill(lNCand);
  }
};

struct cascadefinderQA {
  //Basic checks
  OutputObj<TH3F> h3dMassXiMinus{TH3F("h3dMassXiMinus", "", 20, 0, 100, 200, 0, 10, 200, 1.322 - 0.100, 1.322 + 0.100)};
  OutputObj<TH3F> h3dMassXiPlus{TH3F("h3dMassXiPlus", "", 20, 0, 100, 200, 0, 10, 200, 1.322 - 0.100, 1.322 + 0.100)};
  OutputObj<TH3F> h3dMassOmegaMinus{TH3F("h3dMassOmegaMinus", "", 20, 0, 100, 200, 0, 10, 200, 1.672 - 0.100, 1.672 + 0.100)};
  OutputObj<TH3F> h3dMassOmegaPlus{TH3F("h3dMassOmegaPlus", "", 20, 0, 100, 200, 0, 10, 200, 1.672 - 0.100, 1.672 + 0.100)};

  //Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.999, "V0 CosPA"};       //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<double> casccospa{"casccospa", 0.999, "Casc CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
  Configurable<float> dcacascdau{"dcacascdau", .3, "DCA Casc Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<float> dcabachtopv{"dcabachtopv", .1, "DCA Bach To PV"};
  Configurable<float> dcav0topv{"dcav0topv", .1, "DCA V0 To PV"};
  Configurable<float> v0radius{"v0radius", 2.0, "v0radius"};
  Configurable<float> cascradius{"cascradius", 1.0, "cascradius"};
  Configurable<float> v0masswindow{"v0masswindow", 0.008, "v0masswindow"};

  Filter preFilterV0 =
    aod::cascdata::dcapostopv > dcapostopv&& aod::cascdata::dcanegtopv > dcanegtopv&&
                                                                           aod::cascdata::dcabachtopv > dcabachtopv&&
                                                                                                          aod::cascdata::dcaV0daughters < dcav0dau&& aod::cascdata::dcacascdaughters < dcacascdau;

  ///Connect to CascFinderData: newly indexed, note: CascDataExt table incompatible with standard V0 table!
  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator const& collision, soa::Filtered<aod::CascDataExt> const& Cascades)
  {
    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }
    for (auto& casc : Cascades) {
      //FIXME: dynamic columns cannot be filtered on?
      if (casc.v0radius() > v0radius &&
          casc.cascradius() > cascradius &&
          casc.v0cosPA(collision.posX(), collision.posY(), collision.posZ()) > v0cospa &&
          casc.casccosPA(collision.posX(), collision.posY(), collision.posZ()) > casccospa &&
          casc.dcav0topv(collision.posX(), collision.posY(), collision.posZ()) > dcav0topv) {
        if (casc.charge() < 0) { //FIXME: could be done better...
          if (TMath::Abs(casc.yXi()) < 0.5) {
            h3dMassXiMinus->Fill(collision.centV0M(), casc.pt(), casc.mXi());
          }
          if (TMath::Abs(casc.yOmega()) < 0.5) {
            h3dMassOmegaMinus->Fill(collision.centV0M(), casc.pt(), casc.mOmega());
          }
        } else {
          if (TMath::Abs(casc.yXi()) < 0.5) {
            h3dMassXiPlus->Fill(collision.centV0M(), casc.pt(), casc.mXi());
          }
          if (TMath::Abs(casc.yOmega()) < 0.5) {
            h3dMassOmegaPlus->Fill(collision.centV0M(), casc.pt(), casc.mOmega());
          }
        }
      }
    }
  }
};

/// Extends the cascdata table with expression columns
struct cascadeinitializer {
  Spawns<aod::CascDataExt> cascdataext;
  void init(InitContext const&) {}
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<cascadeprefilter>("lf-cascadeprefilter"),
    adaptAnalysisTask<cascadefinder>("lf-cascadefinder"),
    adaptAnalysisTask<cascadefinderQA>("lf-cascadefinderQA"),
    adaptAnalysisTask<cascadeinitializer>("lf-cascadeinitializer")};
}
