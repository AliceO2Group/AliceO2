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
// Cascade builder task
// =====================
//
// This task loops over an *existing* list of cascades (V0+bachelor track
// indices) and calculates the corresponding full cascade information
//
// Any analysis should loop over the "CascData"
// table as that table contains all information
//
// WARNING: adding filters to the builder IS NOT
// equivalent to re-running the finders. This will only
// ever produce *tighter* selection sections. It is your
// responsibility to check if, by setting a loose filter
// setting, you are going into a region in which no
// candidates exist because the original indices were generated
// using tigher selections.
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
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/StrangenessTables.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

namespace o2::aod
{

namespace cascgood
{
DECLARE_SOA_INDEX_COLUMN_FULL(V0, v0, int, V0DataExt, "fV0Id");
DECLARE_SOA_INDEX_COLUMN_FULL(Bachelor, bachelor, int, FullTracks, "fTracksID");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
} // namespace cascgood
DECLARE_SOA_TABLE(CascGood, "AOD", "CASCGOOD", o2::soa::Index<>,
                  cascgood::V0Id, cascgood::BachelorId,
                  cascgood::CollisionId);
} // namespace o2::aod

using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtended>;

//This prefilter creates a skimmed list of good V0s to re-reconstruct so that
//CPU is saved in case there are specific selections that are to be done
//Note: more configurables, more options to be added as needed
struct cascadeprefilterpairs {
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<float> dcav0topv{"dcav0topv", .1, "DCA V0 To PV"};
  Configurable<double> cospaV0{"cospaV0", .98, "CosPA V0"};
  Configurable<float> lambdamasswindow{"lambdamasswindow", .006, "Distance from Lambda mass"};
  Configurable<float> dcav0dau{"dcav0dau", .6, "DCA V0 Daughters"};
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<float> dcabachtopv{"dcabachtopv", .1, "DCA Bach To PV"};
  Configurable<bool> tpcrefit{"tpcrefit", 1, "demand TPC refit"};
  Configurable<double> v0radius{"v0radius", 0.9, "v0radius"};

  Produces<aod::CascGood> cascgood;
  void process(aod::Collision const& collision, aod::V0DataExt const& V0s, aod::Cascades const& Cascades,
               soa::Join<aod::FullTracks, aod::TracksExtended> const& tracks)
  {
    for (auto& casc : Cascades) {
      //Single-track properties filter
      if (tpcrefit) {
        if (!(casc.v0_as<o2::aod::V0DataExt>().posTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
        if (!(casc.v0_as<o2::aod::V0DataExt>().negTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
        if (!(casc.bachelor_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
      }
      if (casc.v0_as<o2::aod::V0DataExt>().posTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      if (casc.v0_as<o2::aod::V0DataExt>().negTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      if (casc.bachelor_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      if (casc.v0_as<o2::aod::V0DataExt>().posTrack_as<FullTracksExt>().dcaXY() < dcapostopv) {
        continue;
      }
      if (casc.v0_as<o2::aod::V0DataExt>().negTrack_as<FullTracksExt>().dcaXY() < dcanegtopv) {
        continue;
      }
      if (casc.bachelor_as<FullTracksExt>().dcaXY() < dcabachtopv) {
        continue;
      }

      //V0 selections
      if (fabs(casc.v0_as<o2::aod::V0DataExt>().mLambda() - 1.116) > lambdamasswindow && fabs(casc.v0_as<o2::aod::V0DataExt>().mAntiLambda() - 1.116) > lambdamasswindow) {
        continue;
      }
      if (casc.v0_as<o2::aod::V0DataExt>().dcaV0daughters() > dcav0dau) {
        continue;
      }
      if (casc.v0_as<o2::aod::V0DataExt>().v0radius() < v0radius) {
        continue;
      }
      if (casc.v0_as<o2::aod::V0DataExt>().v0cosPA(collision.posX(), collision.posY(), collision.posZ()) < cospaV0) {
        continue;
      }
      if (casc.v0_as<o2::aod::V0DataExt>().dcav0topv(collision.posX(), collision.posY(), collision.posZ()) < dcav0topv) {
        continue;
      }
      cascgood(
        casc.v0_as<o2::aod::V0DataExt>().globalIndex(),
        casc.bachelor_as<FullTracksExt>().globalIndex(),
        casc.bachelor_as<FullTracksExt>().collisionId());
    }
  }
};

/// Cascade builder task: rebuilds cascades
struct cascadebuilder {
  Produces<aod::CascData> cascdata;

  OutputObj<TH1F> hEventCounter{TH1F("hEventCounter", "", 1, 0, 1)};
  OutputObj<TH1F> hCascCandidate{TH1F("hCascCandidate", "", 10, 0, 10)};

  //Configurables
  Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
  Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};

  void process(aod::Collision const& collision, aod::V0DataExt const& V0s, aod::CascGood const& Cascades, aod::FullTracks const& tracks)
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

    hEventCounter->Fill(0.5);
    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};

    for (auto& casc : Cascades) {
      auto charge = -1;
      std::array<float, 3> pos = {0.};
      std::array<float, 3> posXi = {0.};
      std::array<float, 3> pvecpos = {0.};
      std::array<float, 3> pvecneg = {0.};
      std::array<float, 3> pvecbach = {0.};

      hCascCandidate->Fill(0.5);

      //Acquire basic tracks
      auto pTrack = getTrackParCov(casc.v0_as<o2::aod::V0DataExt>().posTrack_as<FullTracksExt>());
      auto nTrack = getTrackParCov(casc.v0_as<o2::aod::V0DataExt>().negTrack_as<FullTracksExt>());
      auto bTrack = getTrackParCov(casc.bachelor_as<FullTracksExt>());
      if (casc.bachelor().signed1Pt() > 0) {
        charge = +1;
      }

      int nCand = fitterV0.process(pTrack, nTrack);
      if (nCand != 0) {
        fitterV0.propagateTracksToVertex();
        hCascCandidate->Fill(1.5);
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
        int nCand2 = fitterCasc.process(tV0, bTrack);
        if (nCand2 != 0) {
          fitterCasc.propagateTracksToVertex();
          hCascCandidate->Fill(2.5);
          const auto& cascvtx = fitterCasc.getPCACandidate();
          for (int i = 0; i < 3; i++) {
            posXi[i] = cascvtx[i];
          }
          fitterCasc.getTrack(1).getPxPyPzGlo(pvecbach);
        } //end if cascade recoed
      }   //end if v0 recoed
      //Fill table, please
      cascdata(
        casc.v0_as<o2::aod::V0DataExt>().globalIndex(),
        casc.bachelor_as<FullTracksExt>().globalIndex(),
        casc.bachelor_as<FullTracksExt>().collisionId(),
        charge, posXi[0], posXi[1], posXi[2], pos[0], pos[1], pos[2],
        pvecpos[0], pvecpos[1], pvecpos[2],
        pvecneg[0], pvecneg[1], pvecneg[2],
        pvecbach[0], pvecbach[1], pvecbach[2],
        fitterV0.getChi2AtPCACandidate(), fitterCasc.getChi2AtPCACandidate(),
        casc.v0().posTrack_as<FullTracksExt>().dcaXY(),
        casc.v0().negTrack_as<FullTracksExt>().dcaXY(),
        casc.bachelor_as<FullTracksExt>().dcaXY());
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
    adaptAnalysisTask<cascadeprefilterpairs>("lf-cascadeprefilterpairs"),
    adaptAnalysisTask<cascadebuilder>("lf-cascadebuilder"),
    adaptAnalysisTask<cascadeinitializer>("lf-cascadeinitializer")};
}
