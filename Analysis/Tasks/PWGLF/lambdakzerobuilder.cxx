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
DECLARE_SOA_INDEX_COLUMN_FULL(PosTrack, posTrack, int, Tracks, "_Pos");
DECLARE_SOA_INDEX_COLUMN_FULL(NegTrack, negTrack, int, Tracks, "_Neg");
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
} // namespace v0goodindices
DECLARE_SOA_TABLE(V0GoodIndices, "AOD", "V0GOODINDICES", o2::soa::Index<>,
                  v0goodindices::PosTrackId, v0goodindices::NegTrackId, v0goodindices::CollisionId);
} // namespace o2::aod

using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtended>;
//using BigTracks = soa::Join<aod::FullTracks, aod::TracksExtended>;
using BigTracks = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended>;
using BigTracksMC = soa::Join<BigTracks, aod::McTrackLabels>;

//This prefilter creates a skimmed list of good V0s to re-reconstruct so that
//CPU is saved in case there are specific selections that are to be done
//Note: more configurables, more options to be added as needed
struct lambdakzeroprefilterpairs {
  Configurable<float> dcanegtopv{"dcanegtopv", .1, "DCA Neg To PV"};
  Configurable<float> dcapostopv{"dcapostopv", .1, "DCA Pos To PV"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<int> tpcrefit{"tpcrefit", 1, "demand TPC refit"};

  HistogramRegistry registry{
    "registry",
    {
      {"hGoodIndices", "hGoodIndices", {HistType::kTH1F, {{4, 0.0f, 4.0f}}}},
    },
  };

  Produces<aod::V0GoodIndices> v0goodindices;

  void process(aod::Collision const& collision, aod::V0s const& V0s,
               BigTracksMC const& tracks, aod::McParticles const& particlesMC)
  {
    for (auto& V0 : V0s) {

      auto labelPos = V0.posTrack_as<BigTracksMC>().label().globalIndex();
      auto labelNeg = V0.negTrack_as<BigTracksMC>().label().globalIndex();
      
      bool isK0SfromLc = (labelPos ==  27384 && labelNeg == 27385) || (labelPos == 981525 && labelNeg == 981526) || (labelPos == 1080007 && labelNeg == 1080008) || (labelPos == 1151717 && labelNeg == 1151718) ||
	(labelPos == 1354080 && labelNeg == 1354081) || (labelPos == 46082 && labelNeg == 46083) || (labelPos == 386994 && labelNeg == 386995) || (labelPos == 1032304 && labelNeg == 1032305) ||
	(labelPos == 1126617 && labelNeg == 1126618) || (labelPos == 1178152 && labelNeg == 1178153) || (labelPos == 1386973 && labelNeg == 1386974) || (labelPos == 18895 && labelNeg == 18896) ||
	(labelPos == 319531 && labelNeg == 319532) || (labelPos == 433387 && labelNeg == 433388) || (labelPos == 914299 && labelNeg == 914300) || (labelPos == 364270 && labelNeg == 364271) ||
	(labelPos == 922284 && labelNeg == 922285) || (labelPos == 49092 && labelNeg == 49093) || (labelPos == 841344 && labelNeg == 841345) || (labelPos == 1167214 && labelNeg == 1167215) ||
	(labelPos == 1257925 && labelNeg == 1257926) || (labelPos == 367299 && labelNeg == 367300) || (labelPos == 439094 && labelNeg == 439095) || (labelPos == 812984 && labelNeg == 812985) ||
	(labelPos == 1379705 && labelNeg == 1379706) || (labelPos == 62529 && labelNeg == 62530) || (labelPos == 299343 && labelNeg == 299344) || (labelPos == 492703 && labelNeg == 492704) ||
	(labelPos == 492681 && labelNeg == 492682) || (labelPos == 540846 && labelNeg == 540847) || (labelPos == 727710 && labelNeg == 727711) || (labelPos == 900248 && labelNeg == 900249) ||
	(labelPos == 653535 && labelNeg == 653536) || (labelPos == 759443 && labelNeg == 759444) || (labelPos == 192861 && labelNeg == 192862) || (labelPos == 1096815 && labelNeg == 1096816) ||
	(labelPos == 1373004 && labelNeg == 1373005) || (labelPos == 62878 && labelNeg == 62879) || (labelPos == 161866 && labelNeg == 161867) || (labelPos == 534341 && labelNeg == 534342) ||
	(labelPos == 806053 && labelNeg == 806054) || (labelPos == 1050897 && labelNeg == 1050898) || (labelPos == 1390049 && labelNeg == 1390050) || (labelPos == 6288 && labelNeg == 6289) ||
	(labelPos == 854422 && labelNeg == 854423) || (labelPos == 576590 && labelNeg == 576591) || (labelPos == 633388 && labelNeg == 633389) || (labelPos == 911572 && labelNeg == 911573) ||
	(labelPos == 995382 && labelNeg == 995383) || (labelPos == 119206 && labelNeg == 119207) || (labelPos == 725047 && labelNeg == 725048) || (labelPos == 762521 && labelNeg == 762522);

	//LOG(INFO) << "posTrackId = " << V0.posTrackId() << ", negTrackId = " << V0.negTrackId();
      
      //Printf("posTrack --> V0.posTrack_as<aod::BigTracksMC>().label().globalIndex() = %ld, negTrack -->  V0.negTrack_as<aod::BigTracksMC>().label().globalIndex() = %ld", V0.posTrack_as<BigTracksMC>().label().globalIndex(), V0.negTrack_as<BigTracksMC>().label().globalIndex());

      if (isK0SfromLc) {
	LOG(INFO) << "V0 builder: found K0S from Lc, posTrack --> " << labelPos << ", negTrack --> " << labelNeg;
      }

      //LOG(INFO) << "posTrack --> V0.posTrack_as<BigTracksMC>().globalIndex() = " << V0.posTrack_as<BigTracksMC>().globalIndex();
      //LOG(INFO) << "posTrack --> V0.posTrack_as<BigTracksMC>().labelId() = " << V0.posTrack_as<BigTracksMC>().labelId();

      //int iiiii = V0.posTrack_as<BigTracksMC>().label().globalIndex();
      //int64_t iiiiiii = V0.posTrack_as<BigTracksMC>().label().globalIndex();
      //int32_t iii = V0.posTrack_as<BigTracksMC>().label().globalIndex();

      //      Printf("V0.posTrack_as<BigTracksMC>().label().globalIndex() as int = %d", iiiii);
      //Printf("V0.posTrack_as<BigTracksMC>().label().globalIndex() as int64_t = %ld", iiiiiii);
      //Printf("V0.posTrack_as<BigTracksMC>().label().globalIndex() as int32_t = %d", iii);

     
      registry.fill(HIST("hGoodIndices"), 0.5);
      if (tpcrefit) {
        if (!(V0.posTrack_as<BigTracksMC>().trackType() & o2::aod::track::TPCrefit)) {
	  if (isK0SfromLc) {
	    LOG(INFO) << "posTrack " << labelPos << " has no TPC refit";
	  }
          continue; //TPC refit
        }
        if (!(V0.negTrack_as<BigTracksMC>().trackType() & o2::aod::track::TPCrefit)) {
	  if (isK0SfromLc) {
	    LOG(INFO) << "negTrack " << labelNeg << " has no TPC refit";
	  }
          continue; //TPC refit
        }
      }
      registry.fill(HIST("hGoodIndices"), 1.5);
      if (V0.posTrack_as<BigTracksMC>().tpcNClsCrossedRows() < mincrossedrows) {
	if (isK0SfromLc) {
	  LOG(INFO) << "posTrack " << labelPos << " has " << V0.posTrack_as<BigTracksMC>().tpcNClsCrossedRows() << " crossed rows, cut at " << mincrossedrows;
	}
        continue;
      }
      if (V0.negTrack_as<BigTracksMC>().tpcNClsCrossedRows() < mincrossedrows) {
	if (isK0SfromLc) {
	  LOG(INFO) << "negTrack " << labelNeg << " has " << V0.negTrack_as<BigTracksMC>().tpcNClsCrossedRows() << " crossed rows, cut at " << mincrossedrows;
	}
	continue;
      }
      registry.fill(HIST("hGoodIndices"), 2.5);
      if (fabs(V0.posTrack_as<BigTracksMC>().dcaXY()) < dcapostopv) {
	if (isK0SfromLc) {
	  LOG(INFO) << "posTrack " << labelPos << " has dcaXY " <<  V0.posTrack_as<BigTracksMC>().dcaXY() << " , cut at " << dcanegtopv;
        }
        continue;
      }
      if (fabs(V0.negTrack_as<BigTracksMC>().dcaXY()) < dcanegtopv) {
	if (isK0SfromLc) {
	  LOG(INFO) << "negTrack " << labelNeg << " has dcaXY " <<  V0.negTrack_as<BigTracksMC>().dcaXY() << " , cut at " << dcanegtopv;
        continue;
	}
      }
      if (isK0SfromLc) {
	LOG(INFO) << "Filling good indices: posTrack --> " << labelPos << ", negTrack --> " << labelNeg;
      }
      registry.fill(HIST("hGoodIndices"), 3.5);
      v0goodindices(V0.posTrack_as<BigTracksMC>().globalIndex(),
		    V0.negTrack_as<BigTracksMC>().globalIndex(),
		    V0.posTrack_as<BigTracksMC>().collisionId());
    
    }
  }
};

/// Cascade builder task: rebuilds cascades
struct lambdakzerobuilder {

  Produces<aod::StoredV0Datas> v0data;

  HistogramRegistry registry{
    "registry",
    {
      {"hEventCounter", "hEventCounter", {HistType::kTH1F, {{1, 0.0f, 1.0f}}}},
      {"hV0Candidate", "hV0Candidate", {HistType::kTH1F, {{2, 0.0f, 2.0f}}}},
    },
  };

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

  /*
  using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtended>;
  using BigTracks = soa::Join<aod::FullTracks, aod::TracksExtended>;
  using BigTracksMC = soa::Join<FullTracksExt, aod::McTrackLabels>;
  */  
  //void process(aod::Collision const& collision, aod::V0GoodIndices const& V0s, soa::Join<aod::FullTracks, aod::TracksExtended> const& tracks, aod::McParticles const& particlesMC)
  void process(aod::Collision const& collision, aod::V0GoodIndices const& V0s, BigTracksMC const& tracks, aod::McParticles const& particlesMC)
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

    registry.fill(HIST("hEventCounter"), 0.5);
    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};

    for (auto& V0 : V0s) {
      std::array<float, 3> pos = {0.};
      std::array<float, 3> pvec0 = {0.};
      std::array<float, 3> pvec1 = {0.};

      auto labelPos = V0.posTrack_as<BigTracksMC>().label().globalIndex();
      auto labelNeg = V0.negTrack_as<BigTracksMC>().label().globalIndex();
      
      bool isK0SfromLc = (labelPos ==  27384 && labelNeg == 27385) || (labelPos == 981525 && labelNeg == 981526) || (labelPos == 1080007 && labelNeg == 1080008) || (labelPos == 1151717 && labelNeg == 1151718) ||
	(labelPos == 1354080 && labelNeg == 1354081) || (labelPos == 46082 && labelNeg == 46083) || (labelPos == 386994 && labelNeg == 386995) || (labelPos == 1032304 && labelNeg == 1032305) ||
	(labelPos == 1126617 && labelNeg == 1126618) || (labelPos == 1178152 && labelNeg == 1178153) || (labelPos == 1386973 && labelNeg == 1386974) || (labelPos == 18895 && labelNeg == 18896) ||
	(labelPos == 319531 && labelNeg == 319532) || (labelPos == 433387 && labelNeg == 433388) || (labelPos == 914299 && labelNeg == 914300) || (labelPos == 364270 && labelNeg == 364271) ||
	(labelPos == 922284 && labelNeg == 922285) || (labelPos == 49092 && labelNeg == 49093) || (labelPos == 841344 && labelNeg == 841345) || (labelPos == 1167214 && labelNeg == 1167215) ||
	(labelPos == 1257925 && labelNeg == 1257926) || (labelPos == 367299 && labelNeg == 367300) || (labelPos == 439094 && labelNeg == 439095) || (labelPos == 812984 && labelNeg == 812985) ||
	(labelPos == 1379705 && labelNeg == 1379706) || (labelPos == 62529 && labelNeg == 62530) || (labelPos == 299343 && labelNeg == 299344) || (labelPos == 492703 && labelNeg == 492704) ||
	(labelPos == 492681 && labelNeg == 492682) || (labelPos == 540846 && labelNeg == 540847) || (labelPos == 727710 && labelNeg == 727711) || (labelPos == 900248 && labelNeg == 900249) ||
	(labelPos == 653535 && labelNeg == 653536) || (labelPos == 759443 && labelNeg == 759444) || (labelPos == 192861 && labelNeg == 192862) || (labelPos == 1096815 && labelNeg == 1096816) ||
	(labelPos == 1373004 && labelNeg == 1373005) || (labelPos == 62878 && labelNeg == 62879) || (labelPos == 161866 && labelNeg == 161867) || (labelPos == 534341 && labelNeg == 534342) ||
	(labelPos == 806053 && labelNeg == 806054) || (labelPos == 1050897 && labelNeg == 1050898) || (labelPos == 1390049 && labelNeg == 1390050) || (labelPos == 6288 && labelNeg == 6289) ||
	(labelPos == 854422 && labelNeg == 854423) || (labelPos == 576590 && labelNeg == 576591) || (labelPos == 633388 && labelNeg == 633389) || (labelPos == 911572 && labelNeg == 911573) ||
	(labelPos == 995382 && labelNeg == 995383) || (labelPos == 119206 && labelNeg == 119207) || (labelPos == 725047 && labelNeg == 725048) || (labelPos == 762521 && labelNeg == 762522);

      registry.fill(HIST("hV0Candidate"), 0.5);

      auto pTrack = getTrackParCov(V0.posTrack_as<BigTracksMC>());
      auto nTrack = getTrackParCov(V0.negTrack_as<BigTracksMC>());
      int nCand = fitter.process(pTrack, nTrack);
      if (isK0SfromLc) {
	LOG(INFO) << "fitter: nCand = " << nCand << " for posTrack --> " << labelPos << ", negTrack --> " << labelNeg;
      }
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

      if (isK0SfromLc) {
	LOG(INFO) << "in builder 0: posTrack --> " << labelPos << ", negTrack --> " << labelNeg;
      }

      //Apply selections so a skimmed table is created only
      if (fitter.getChi2AtPCACandidate() > dcav0dau) {
	if (isK0SfromLc) {
	  LOG(INFO) << "posTrack --> " << labelPos << ", negTrack --> " << labelNeg << " will be skipped due to dca cut";
	}
        continue;
      }

      auto V0CosinePA = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, array{pos[0], pos[1], pos[2]}, array{pvec0[0] + pvec1[0], pvec0[1] + pvec1[1], pvec0[2] + pvec1[2]});
      if (V0CosinePA < v0cospa) {
	if (isK0SfromLc) {
	  LOG(INFO) << "posTrack --> " << labelPos << ", negTrack --> " << labelNeg << " will be skipped due to CPA cut";
	}
        continue;
      }

      auto V0radius = RecoDecay::sqrtSumOfSquares(pos[0], pos[1]); //probably find better name to differentiate the cut from the variable
      //LOG(INFO) << "V0radius of the candidate = " << V0radius << ", cut = " << v0radius;
      if (V0radius < v0radius) {
	if (isK0SfromLc) {
	  LOG(INFO) << "posTrack --> " << labelPos << ", negTrack --> " << labelNeg << " will be skipped due to radius cut";
	}
        continue;
      }

      if (isK0SfromLc) {
	LOG(INFO) << "in builder 1, keeping K0S candidate: posTrack --> " << labelPos << ", negTrack --> " << labelNeg;
      }
      
      registry.fill(HIST("hV0Candidate"), 1.5);
      v0data(
        V0.posTrack_as<BigTracksMC>().globalIndex(),
        V0.negTrack_as<BigTracksMC>().globalIndex(),
        V0.negTrack_as<BigTracksMC>().collisionId(),
        fitter.getTrack(0).getX(), fitter.getTrack(1).getX(),
        pos[0], pos[1], pos[2],
        pvec0[0], pvec0[1], pvec0[2],
        pvec1[0], pvec1[1], pvec1[2],
        fitter.getChi2AtPCACandidate(),
        V0.posTrack_as<BigTracksMC>().dcaXY(),
        V0.negTrack_as<BigTracksMC>().dcaXY());
    }
  }
};

/// Extends the v0data table with expression columns
struct lambdakzeroinitializer {
  Spawns<aod::V0Datas> v0datas;
  void init(InitContext const&) {}
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<lambdakzeroprefilterpairs>(cfgc, TaskName{"lf-lambdakzeroprefilterpairs"}),
    adaptAnalysisTask<lambdakzerobuilder>(cfgc, TaskName{"lf-lambdakzerobuilder"}),
    adaptAnalysisTask<lambdakzeroinitializer>(cfgc, TaskName{"lf-lambdakzeroinitializer"})};
}
