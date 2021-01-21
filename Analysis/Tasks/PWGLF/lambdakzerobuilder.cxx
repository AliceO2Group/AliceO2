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
// V0 builder task
// ================
//
// This task loops over an *existing* list of V0s (neg/pos track
// indices) and calculates the corresponding full V0 information
//
// Any analysis should loop over the "V0Data"
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
#include "AnalysisDataModel/StrangenessTables.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

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

//This table stores a filtered list of valid V0 indices
namespace o2::aod
{
namespace v0goodindices
{
DECLARE_SOA_INDEX_COLUMN_FULL(PosTrack, posTrack, int, FullTracks, "fPositiveTrackID");
DECLARE_SOA_INDEX_COLUMN_FULL(NegTrack, negTrack, int, FullTracks, "fNegativeTrackID");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
} // namespace v0goodindices
DECLARE_SOA_TABLE(V0GoodIndices, "AOD", "V0GOODINDICES", o2::soa::Index<>,
                  v0goodindices::PosTrackId, v0goodindices::NegTrackId, v0goodindices::CollisionId);
} // namespace o2::aod

using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtended>;

//This prefilter creates a skimmed list of good V0s to re-reconstruct so that
//CPU is saved in case there are specific selections that are to be done
//Note: more configurables, more options to be added as needed
struct lambdakzeroprefilterpairs {
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<int> tpcrefit{"tpcrefit", 1, "demand TPC refit"};

  OutputObj<TH1F> hGoodIndices{TH1F("hGoodIndices", "", 4, 0, 4)};

  Produces<aod::V0GoodIndices> v0goodindices;

  void process(aod::Collision const& collision, aod::V0s const& V0s,
               soa::Join<aod::FullTracks, aod::TracksExtended> const& tracks)
  {
    for (auto& V0 : V0s) {
      hGoodIndices->Fill(0.5);
      if (tpcrefit) {
        if (!(V0.posTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
        if (!(V0.negTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
      }
      hGoodIndices->Fill(1.5);
      if (V0.posTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      if (V0.negTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      hGoodIndices->Fill(2.5);
      if (fabs(V0.posTrack_as<FullTracksExt>().dcaXY()) < dcapostopv) {
        continue;
      }
      if (fabs(V0.negTrack_as<FullTracksExt>().dcaXY()) < dcanegtopv) {
        continue;
      }
      hGoodIndices->Fill(3.5);
      v0goodindices(V0.posTrack_as<FullTracksExt>().globalIndex(),
                    V0.negTrack_as<FullTracksExt>().globalIndex(),
                    V0.posTrack_as<FullTracksExt>().collisionId());
    }
  }
};

/// Cascade builder task: rebuilds cascades
struct lambdakzerobuilder {

  Produces<aod::V0Data> v0data;

  OutputObj<TH1F> hEventCounter{TH1F("hEventCounter", "", 1, 0, 1)};
  OutputObj<TH1F> hV0Candidate{TH1F("hV0Candidate", "", 2, 0, 2)};

  //Configurables
  Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
  // Configurable<int> d_UseAbsDCA{"d_UseAbsDCA", 1, "Use Abs DCAs"}; uncomment this once we want to use the weighted DCA

  //Selection criteria
  Configurable<double> v0cospa{"v0cospa", 0.995, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 1.0, "DCA V0 Daughters"};
  Configurable<float> v0radius{"v0radius", 5.0, "v0radius"};

  double massPi = TDatabasePDG::Instance()->GetParticle(kPiPlus)->Mass();
  double massKa = TDatabasePDG::Instance()->GetParticle(kKPlus)->Mass();
  double massPr = TDatabasePDG::Instance()->GetParticle(kProton)->Mass();

  using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtended>;

  void process(aod::Collision const& collision, aod::V0GoodIndices const& V0s, soa::Join<aod::FullTracks, aod::TracksExtended> const& tracks)
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
    fitter.setUseAbsDCA(true); // use d_UseAbsDCA once we want to use the weighted DCA

    hEventCounter->Fill(0.5);
    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};

    for (auto& V0 : V0s) {
      std::array<float, 3> pos = {0.};
      std::array<float, 3> pvec0 = {0.};
      std::array<float, 3> pvec1 = {0.};

      hV0Candidate->Fill(0.5);

      auto pTrack = getTrackParCov(V0.posTrack_as<FullTracksExt>());
      auto nTrack = getTrackParCov(V0.negTrack_as<FullTracksExt>());
      int nCand = fitter.process(pTrack, nTrack);
      if (nCand != 0) {
        fitter.propagateTracksToVertex();
        const auto& vtx = fitter.getPCACandidate();
        for (int i = 0; i < 3; i++) {
          pos[i] = vtx[i];
        }
        fitter.getTrack(0).getPxPyPzGlo(pvec0);
        fitter.getTrack(1).getPxPyPzGlo(pvec1);
      } else {
        continue;
      }

      //Apply selections so a skimmed table is created only
      if (fitter.getChi2AtPCACandidate() > dcav0dau) {
        continue;
      }

      auto V0CosinePA = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, array{pos[0], pos[1], pos[2]}, array{pvec0[0] + pvec1[0], pvec0[1] + pvec1[1], pvec0[2] + pvec1[2]});
      if (V0CosinePA < v0cospa) {
        continue;
      }

      auto V0radius = RecoDecay::sqrtSumOfSquares(pos[0], pos[1]); //probably find better name to differentiate the cut from the variable
      if (V0radius < v0radius) {
        continue;
      }

      hV0Candidate->Fill(1.5);
      v0data(
        V0.posTrack_as<FullTracksExt>().globalIndex(),
        V0.negTrack_as<FullTracksExt>().globalIndex(),
        V0.negTrack_as<FullTracksExt>().collisionId(),
        fitter.getTrack(0).getX(), fitter.getTrack(1).getX(),
        pos[0], pos[1], pos[2],
        pvec0[0], pvec0[1], pvec0[2],
        pvec1[0], pvec1[1], pvec1[2],
        fitter.getChi2AtPCACandidate(),
        V0.posTrack_as<FullTracksExt>().dcaXY(),
        V0.negTrack_as<FullTracksExt>().dcaXY());
    }
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
    adaptAnalysisTask<lambdakzeroprefilterpairs>("lf-lambdakzeroprefilterpairs"),
    adaptAnalysisTask<lambdakzerobuilder>("lf-lambdakzerobuilder"),
    adaptAnalysisTask<lambdakzeroinitializer>("lf-lambdakzeroinitializer")};
}
