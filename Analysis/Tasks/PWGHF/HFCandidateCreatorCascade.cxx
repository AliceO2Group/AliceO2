// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFCandidateCreatorCascade.cxx
/// \brief Reconstruction of heavy-flavour cascade decay candidates
///

#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/V0.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Perform MC matching."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// Reconstruction of heavy-flavour cascade decay candidates
struct HFCandidateCreatorCascade {

  Produces<aod::HfCandCascBase> rowCandidateBase;

  Configurable<double> d_bZ{"d_bZ", 5., "magnetic field"};
  Configurable<bool> b_propDCA{"b_propDCA", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxR{"d_maxR", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxDZIni{"d_maxDZIni", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minParamChange{"d_minParamChange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minRelChi2Change{"d_minRelChi2Change", 0.9, "stop iterations is chi2/chi2old > this"};
  Configurable<bool> b_doValPlots{"b_doValPlots", true, "do validation plots"};

  OutputObj<TH1F> hmass2{TH1F("hmass2", "2-prong candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", 500, 0., 5.)};
  OutputObj<TH1F> hCovPVXX{TH1F("hCovPVXX", "2-prong candidates;XX element of cov. matrix of prim. vtx. position (cm^{2});entries", 100, 0., 1.e-4)};
  OutputObj<TH1F> hCovSVXX{TH1F("hCovSVXX", "2-prong candidates;XX element of cov. matrix of sec. vtx. position (cm^{2});entries", 100, 0., 0.2)};

  double massP = RecoDecay::getMassPDG(kProton);
  double massK0s = RecoDecay::getMassPDG(kK0Short);
  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massLc = RecoDecay::getMassPDG(4122);
  double mass2K0sP{0.};

  using MyTracks = aod::BigTracksMC;
  //using MyTracks = aod::BigTracks;

  void process(aod::Collisions const& collisions,
               aod::HfTrackIndexCasc const& rowsTrackIndexCasc,
               MyTracks const& tracks,
               aod::V0DataExt const& V0s,
	       aod::McParticles& mcParticles)
  {
    // 2-prong vertex fitter
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(d_bZ);
    df.setPropagateToPCA(b_propDCA);
    df.setMaxR(d_maxR);
    df.setMaxDZIni(d_maxDZIni);
    df.setMinParamChange(d_minParamChange);
    df.setMinRelChi2Change(d_minRelChi2Change);
    df.setUseAbsDCA(true);

    // loop over pairs of track indeces
    for (const auto& casc : rowsTrackIndexCasc) {
      const auto& bach = casc.index0_as<MyTracks>();
      auto trackParCovBach = getTrackParCov(bach);
      const auto& v0 = casc.indexV0_as<o2::aod::V0DataExt>();
      const auto& posTrack = v0.posTrack_as<MyTracks>();
      const auto& negTrack = v0.negTrack_as<MyTracks>();

      auto protonLabel = bach.labelId();
      auto labelPos = posTrack.labelId();
      auto labelNeg = negTrack.labelId();

      bool isLc = (protonLabel == 27378 && labelPos ==  27384 && labelNeg == 27385) || (protonLabel == 981514 && labelPos == 981525 && labelNeg == 981526) || (protonLabel == 1079941 && labelPos == 1080007 && labelNeg == 1080008) ||
	(protonLabel == 1151713 && labelPos == 1151717 && labelNeg == 1151718) || (protonLabel == 1354075 && labelPos == 1354080 && labelNeg == 1354081) || (protonLabel == 46077 && labelPos == 46082 && labelNeg == 46083) ||
	(protonLabel == 386988 && labelPos == 386994 && labelNeg == 386995) || (protonLabel == 1032251 && labelPos == 1032304 && labelNeg == 1032305) || (protonLabel == 1126614 && labelPos == 1126617 && labelNeg == 1126618) ||
	(protonLabel == 1178107 && labelPos == 1178152 && labelNeg == 1178153) || (protonLabel == 1386970 && labelPos == 1386973 && labelNeg == 1386974) || (protonLabel == 18733 && labelPos == 18895 && labelNeg == 18896) ||
	(protonLabel == 319481 && labelPos == 319531 && labelNeg == 319532) || (protonLabel == 433384 && labelPos == 433387 && labelNeg == 433388) || (protonLabel == 914259 && labelPos == 914299 && labelNeg == 914300) ||
	(protonLabel == 364214 && labelPos == 364270 && labelNeg == 364271) || (protonLabel == 922267 && labelPos == 922284 && labelNeg == 922285) || (protonLabel == 49070 && labelPos == 49092 && labelNeg == 49093) ||
	(protonLabel == 841303 && labelPos == 841344 && labelNeg == 841345) || (protonLabel == 1167211 && labelPos == 1167214 && labelNeg == 1167215) || (protonLabel == 1257919 && labelPos == 1257925 && labelNeg == 1257926) ||
	(protonLabel == 367228 && labelPos == 367299 && labelNeg == 367300) || (protonLabel == 439084 && labelPos == 439094 && labelNeg == 439095) || (protonLabel == 812970 && labelPos == 812984 && labelNeg == 812985) ||
	(protonLabel == 1379678 && labelPos == 1379705 && labelNeg == 1379706) || (protonLabel == 62526 && labelPos == 62529 && labelNeg == 62530) || (protonLabel == 299330 && labelPos == 299343 && labelNeg == 299344) ||
	(protonLabel == 492671 && labelPos == 492703 && labelNeg == 492704) || (protonLabel == 492678 && labelPos == 492681 && labelNeg == 492682) || (protonLabel == 540812 && labelPos == 540846 && labelNeg == 540847) ||
	(protonLabel == 727692 && labelPos == 727710 && labelNeg == 727711) || (protonLabel == 900211 && labelPos == 900248 && labelNeg == 900249) || (protonLabel == 653455 && labelPos == 653535 && labelNeg == 653536) ||
	(protonLabel == 759316 && labelPos == 759443 && labelNeg == 759444) || (protonLabel == 192853 && labelPos == 192861 && labelNeg == 192862) || (protonLabel == 1096808 && labelPos == 1096815 && labelNeg == 1096816) ||
	(protonLabel == 1373001 && labelPos == 1373004 && labelNeg == 1373005) || (protonLabel == 62875 && labelPos == 62878 && labelNeg == 62879) || (protonLabel == 161859 && labelPos == 161866 && labelNeg == 161867) ||
	(protonLabel == 534335 && labelPos == 534341 && labelNeg == 534342) || (protonLabel == 806033 && labelPos == 806053 && labelNeg == 806054) || (protonLabel == 1050891 && labelPos == 1050897 && labelNeg == 1050898) ||
	(protonLabel == 1390046 && labelPos == 1390049 && labelNeg == 1390050) || (protonLabel == 6268 && labelPos == 6288 && labelNeg == 6289) || (protonLabel == 854417 && labelPos == 854422 && labelNeg == 854423) ||
	(protonLabel == 576587 && labelPos == 576590 && labelNeg == 576591) || (protonLabel == 633385 && labelPos == 633388 && labelNeg == 633389) || (protonLabel == 911527 && labelPos == 911572 && labelNeg == 911573) ||
	(protonLabel == 995379 && labelPos == 995382 && labelNeg == 995383) || (protonLabel == 119194 && labelPos == 119206 && labelNeg == 119207) || (protonLabel == 724999 && labelPos == 725047 && labelNeg == 725048) ||
	(protonLabel == 762518 && labelPos == 762521 && labelNeg == 762522);
      
      if (isLc) {
	LOG(INFO) << "Processing the Lc with proton " << protonLabel << " posTrack " << labelPos << " negTrack " << labelNeg;
      }
      
      auto posTrackParCov = getTrackParCov(posTrack); // check that MyTracks does not need TracksExtended!
      auto negTrackParCov = getTrackParCov(negTrack); // check that MyTracks does not need TracksExtended!
      posTrackParCov.propagateTo(v0.posX(), d_bZ);    // propagate the track to the X closest to the V0 vertex
      negTrackParCov.propagateTo(v0.negX(), d_bZ);    // propagate the track to the X closest to the V0 vertex
      const std::array<float, 3> vertexV0 = {v0.x(), v0.y(), v0.z()};
      const std::array<float, 3> momentumV0 = {v0.px(), v0.py(), v0.pz()};
      // we build the neutral track to then build the cascade
      auto trackV0 = o2::dataformats::V0(vertexV0, momentumV0, posTrackParCov, negTrackParCov, {0, 0}, {0, 0}); // build the V0 track (indices for v0 daughters set to 0 for now)

      auto collision = bach.collision();

      // reconstruct the cascade secondary vertex
      if (df.process(trackV0, trackParCovBach) == 0) {
	if (isLc) LOG(INFO) << "Vertexing failed for Lc candidate";
	//	if (isLc) {
	// LOG(INFO) << "Vertexing failed for Lc with proton " << protonLabel << " posTrack " << labelPos << " negTrack " << labelNeg;
	//}
        continue;
      }
      else {
	//LOG(INFO) << "Vertexing succeeded for Lc candidate";
      }

      const auto& secondaryVertex = df.getPCACandidate();
      auto chi2PCA = df.getChi2AtPCACandidate();
      auto covMatrixPCA = df.calcPCACovMatrix().Array();
      hCovSVXX->Fill(covMatrixPCA[0]); // FIXME: Calculation of errorDecayLength(XY) gives wrong values without this line.
      // do I have to call "df.propagateTracksToVertex();"?
      auto trackParVarV0 = df.getTrack(0);
      auto trackParVarBach = df.getTrack(1);

      // get track momenta
      array<float, 3> pVecV0;
      array<float, 3> pVecBach;
      trackParVarV0.getPxPyPzGlo(pVecV0);
      trackParVarBach.getPxPyPzGlo(pVecBach);

      // get track impact parameters
      // This modifies track momenta!
      auto primaryVertex = getPrimaryVertex(collision);
      auto covMatrixPV = primaryVertex.getCov();
      hCovPVXX->Fill(covMatrixPV[0]);
      o2::dataformats::DCA impactParameterV0;
      o2::dataformats::DCA impactParameterBach;
      trackParVarV0.propagateToDCA(primaryVertex, d_bZ, &impactParameterV0); // we do this wrt the primary vtx
      trackParVarBach.propagateToDCA(primaryVertex, d_bZ, &impactParameterBach);

      // get uncertainty of the decay length
      double phi, theta;
      getPointDirection(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex, phi, theta);
      auto errorDecayLength = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, theta) + getRotatedCovMatrixXX(covMatrixPCA, phi, theta));
      auto errorDecayLengthXY = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, 0.) + getRotatedCovMatrixXX(covMatrixPCA, phi, 0.));

      // fill candidate table rows
      if (isLc) {
	LOG(INFO) << "IT IS A Lc! Filling for Lc with proton " << protonLabel << " posTrack " << labelPos << " negTrack " << labelNeg;
      }
      rowCandidateBase(collision.posX(), collision.posY(), collision.posZ(),
                       secondaryVertex[0], secondaryVertex[1], secondaryVertex[2],
                       errorDecayLength, errorDecayLengthXY,
                       chi2PCA,
                       pVecBach[0], pVecBach[1], pVecBach[2],
                       pVecV0[0], pVecV0[1], pVecV0[2],
                       impactParameterBach.getY(), impactParameterV0.getY(),
                       std::sqrt(impactParameterBach.getSigmaY2()), std::sqrt(impactParameterV0.getSigmaY2()),
                       casc.index0Id(), casc.indexV0Id(),
                       casc.hfflag(),
                       v0.x(), v0.y(), v0.z(),
                       //v0.posTrack(), v0.negTrack(), // why this was not fine?
                       v0.posTrack_as<MyTracks>().globalIndex(), v0.negTrack_as<MyTracks>().globalIndex(),
                       v0.pxpos(), v0.pypos(), v0.pzpos(),
                       v0.pxneg(), v0.pyneg(), v0.pzneg(),
                       v0.dcaV0daughters(),
                       v0.dcapostopv(),
                       v0.dcanegtopv());

      // fill histograms
      if (b_doValPlots) {
        // calculate invariant masses
        auto arrayMomenta = array{pVecBach, pVecV0};
        mass2K0sP = RecoDecay::M(arrayMomenta, array{massP, massK0s});
        hmass2->Fill(mass2K0sP);
      }
    }
  }
};

/// Extends the base table with expression columns.
struct HFCandidateCreatorCascadeExpressions {
  Spawns<aod::HfCandCascExt> rowCandidateCasc;
  void init(InitContext const&) {}
};

//___________________________________________________________________________________________

/// Performs MC matching.
struct HFCandidateCreatorCascadeMC {
  Produces<aod::HfCandCascadeMCRec> rowMCMatchRec;
  Produces<aod::HfCandCascadeMCGen> rowMCMatchGen;

  void process(aod::HfCandCascade const& candidates,
               aod::BigTracksMC const& tracks,
               aod::McParticles const& particlesMC)
  {
    int8_t sign = 0;
    std::vector<int> arrDaughLcIndex;
    std::array<int, 3> arrDaughLcPDG;
    std::array<int, 3> arrDaughLcPDGRef = {2212, 211, -211};
    
    // Match reconstructed candidates.
    for (auto& candidate : candidates) {
      auto arrayDaughtersV0 = array{candidate.posTrack_as<aod::BigTracksMC>(), candidate.negTrack_as<aod::BigTracksMC>()};
      auto arrayDaughtersLc = array{candidate.index0_as<aod::BigTracksMC>(), candidate.posTrack_as<aod::BigTracksMC>(), candidate.negTrack_as<aod::BigTracksMC>()};

      // First we check the K0s
      printf("\n");
      LOG(INFO) << "Checking MC for candidate!";
      LOG(DEBUG) << "Looking for K0s";
      auto labelPos = candidate.posTrack_as<aod::BigTracksMC>().label().globalIndex();
      auto labelNeg = candidate.negTrack_as<aod::BigTracksMC>().label().globalIndex();
      auto protonLabel = candidate.index0_as<aod::BigTracksMC>().labelId();
      
      bool isLc = (protonLabel == 27378 && labelPos ==  27384 && labelNeg == 27385) || (protonLabel == 981514 && labelPos == 981525 && labelNeg == 981526) || (protonLabel == 1079941 && labelPos == 1080007 && labelNeg == 1080008) ||
	(protonLabel == 1151713 && labelPos == 1151717 && labelNeg == 1151718) || (protonLabel == 1354075 && labelPos == 1354080 && labelNeg == 1354081) || (protonLabel == 46077 && labelPos == 46082 && labelNeg == 46083) ||
	(protonLabel == 386988 && labelPos == 386994 && labelNeg == 386995) || (protonLabel == 1032251 && labelPos == 1032304 && labelNeg == 1032305) || (protonLabel == 1126614 && labelPos == 1126617 && labelNeg == 1126618) ||
	(protonLabel == 1178107 && labelPos == 1178152 && labelNeg == 1178153) || (protonLabel == 1386970 && labelPos == 1386973 && labelNeg == 1386974) || (protonLabel == 18733 && labelPos == 18895 && labelNeg == 18896) ||
	(protonLabel == 319481 && labelPos == 319531 && labelNeg == 319532) || (protonLabel == 433384 && labelPos == 433387 && labelNeg == 433388) || (protonLabel == 914259 && labelPos == 914299 && labelNeg == 914300) ||
	(protonLabel == 364214 && labelPos == 364270 && labelNeg == 364271) || (protonLabel == 922267 && labelPos == 922284 && labelNeg == 922285) || (protonLabel == 49070 && labelPos == 49092 && labelNeg == 49093) ||
	(protonLabel == 841303 && labelPos == 841344 && labelNeg == 841345) || (protonLabel == 1167211 && labelPos == 1167214 && labelNeg == 1167215) || (protonLabel == 1257919 && labelPos == 1257925 && labelNeg == 1257926) ||
	(protonLabel == 367228 && labelPos == 367299 && labelNeg == 367300) || (protonLabel == 439084 && labelPos == 439094 && labelNeg == 439095) || (protonLabel == 812970 && labelPos == 812984 && labelNeg == 812985) ||
	(protonLabel == 1379678 && labelPos == 1379705 && labelNeg == 1379706) || (protonLabel == 62526 && labelPos == 62529 && labelNeg == 62530) || (protonLabel == 299330 && labelPos == 299343 && labelNeg == 299344) ||
	(protonLabel == 492671 && labelPos == 492703 && labelNeg == 492704) || (protonLabel == 492678 && labelPos == 492681 && labelNeg == 492682) || (protonLabel == 540812 && labelPos == 540846 && labelNeg == 540847) ||
	(protonLabel == 727692 && labelPos == 727710 && labelNeg == 727711) || (protonLabel == 900211 && labelPos == 900248 && labelNeg == 900249) || (protonLabel == 653455 && labelPos == 653535 && labelNeg == 653536) ||
	(protonLabel == 759316 && labelPos == 759443 && labelNeg == 759444) || (protonLabel == 192853 && labelPos == 192861 && labelNeg == 192862) || (protonLabel == 1096808 && labelPos == 1096815 && labelNeg == 1096816) ||
	(protonLabel == 1373001 && labelPos == 1373004 && labelNeg == 1373005) || (protonLabel == 62875 && labelPos == 62878 && labelNeg == 62879) || (protonLabel == 161859 && labelPos == 161866 && labelNeg == 161867) ||
	(protonLabel == 534335 && labelPos == 534341 && labelNeg == 534342) || (protonLabel == 806033 && labelPos == 806053 && labelNeg == 806054) || (protonLabel == 1050891 && labelPos == 1050897 && labelNeg == 1050898) ||
	(protonLabel == 1390046 && labelPos == 1390049 && labelNeg == 1390050) || (protonLabel == 6268 && labelPos == 6288 && labelNeg == 6289) || (protonLabel == 854417 && labelPos == 854422 && labelNeg == 854423) ||
	(protonLabel == 576587 && labelPos == 576590 && labelNeg == 576591) || (protonLabel == 633385 && labelPos == 633388 && labelNeg == 633389) || (protonLabel == 911527 && labelPos == 911572 && labelNeg == 911573) ||
	(protonLabel == 995379 && labelPos == 995382 && labelNeg == 995383) || (protonLabel == 119194 && labelPos == 119206 && labelNeg == 119207) || (protonLabel == 724999 && labelPos == 725047 && labelNeg == 725048) ||
	(protonLabel == 762518 && labelPos == 762521 && labelNeg == 762522);
      //if (labelPos ==  27384 || labelPos == 981525 || labelPos == 1080007 || labelPos == 1151717 || labelPos == 1354080 || labelPos == 46082 || labelPos == 386994 || labelPos == 1032304 || labelPos == 1126617 || labelPos == 1178152 || labelPos == 1386973 || labelPos == 18895 || labelPos == 319531 || labelPos == 433387 || labelPos == 914299 || labelPos == 364270 || labelPos == 922284 || labelPos == 49092 || labelPos == 841344 || labelPos == 1167214 || labelPos == 1257925 || labelPos == 367299 || labelPos == 439094 || labelPos == 812984 || labelPos == 1379705 || labelPos == 62529 || labelPos == 299343 || labelPos == 492703 || labelPos == 492681 || labelPos == 540846 || labelPos == 727710 || labelPos == 900248 || labelPos == 653535 || labelPos == 759443 || labelPos == 192861 || labelPos == 1096815 || labelPos == 1373004 || labelPos == 62878 || labelPos == 161866 || labelPos == 534341 || labelPos == 806053 || labelPos == 1050897 || labelPos == 1390049 || labelPos == 6288 || labelPos == 854422 || labelPos == 576590 || labelPos == 633388 || labelPos == 911572 || labelPos == 995382 || labelPos == 119206 || labelPos == 725047 || labelPos == 762521 ||
      //  labelNeg ==  27385 || labelNeg == 981526 || labelNeg == 1080008 || labelNeg == 1151718 || labelNeg == 1354081 || labelNeg == 46083 || labelNeg == 386995 || labelNeg == 1032305 || labelNeg == 1126618 || labelNeg == 1178153 || labelNeg == 1386974 || labelNeg == 18896 || labelNeg == 319532 || labelNeg == 433388 || labelNeg == 914300 || labelNeg == 364271 || labelNeg == 922285 || labelNeg == 49093 || labelNeg == 841345 || labelNeg == 1167215 || labelNeg == 1257926 || labelNeg == 367300 || labelNeg == 439095 || labelNeg == 812985 || labelNeg == 1379706 || labelNeg == 62530 || labelNeg == 299344 || labelNeg == 492704 || labelNeg == 492682 || labelNeg == 540847 || labelNeg == 727711 || labelNeg == 900249 || labelNeg == 653536 || labelNeg == 759444 || labelNeg == 192862 || labelNeg == 1096816 || labelNeg == 1373005 || labelNeg == 62879 || labelNeg == 161867 || labelNeg == 534342 || labelNeg == 806054 || labelNeg == 1050898 || labelNeg == 1390050 || labelNeg == 6289 || labelNeg == 854423 || labelNeg == 576591 || labelNeg == 633389 || labelNeg == 911573 || labelNeg == 995383 || labelNeg == 119207 || labelNeg == 725048 || labelNeg == 762522){
      if ((labelPos ==  27384 && labelNeg == 27385) || (labelPos == 981525 && labelNeg == 981526) || (labelPos == 1080007 && labelNeg == 1080008) || (labelPos == 1151717 && labelNeg == 1151718) ||
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
	  (labelPos == 995382 && labelNeg == 995383) || (labelPos == 119206 && labelNeg == 119207) || (labelPos == 725047 && labelNeg == 725048) || (labelPos == 762521 && labelNeg == 762522)) {
	LOG(INFO) << "correct K0S in the Lc daughters: posTrack --> " << labelPos << ", negTrack --> " << labelNeg;
      }

      //if (isLc) {
	RecoDecay::getMatchedMCRec(particlesMC, arrayDaughtersV0, 310, array{+kPiPlus, -kPiPlus}, true, &sign, 1); // does it matter the "acceptAntiParticle" in the K0s case? In principle, there is no anti-K0s
	
	if (sign != 0) {                                                                                           // we have already positively checked the K0s
	  // then we check the Lc
	  LOG(INFO) << "K0S was correct! now we check the Lc";
	  auto labelProton = candidate.index0_as<aod::BigTracksMC>().label().globalIndex();
	  LOG(INFO) << "label proton = " << labelProton;
	  LOG(DEBUG) << "Looking for Lc";
	  RecoDecay::getMatchedMCRec(particlesMC, arrayDaughtersLc, 4122, array{+kProton, +kPiPlus, -kPiPlus}, true, &sign, 3); // 3-levels Lc --> p + K0 --> p + K0s --> p + pi+ pi-
	  if (sign == 0) {
	    LOG(INFO) << "No true Lc found";
	  }
	  else {
	    LOG(INFO) << "Lc found with sign " << sign;
	  }
	  printf("\n");
	}
	
	rowMCMatchRec(sign);
      }
    //}
    
    // Match generated particles.
    for (auto& particle : particlesMC) {
      // checking if I have a Lc --> K0S + p
      RecoDecay::isMatchedMCGen(particlesMC, particle, 4122, array{+kProton, 310}, true, &sign, 2);
      if (sign != 0) {
	LOG(INFO) << "Lc in K0S p";
	arrDaughLcIndex.clear();
	// checking that the final daughters (decay depth = 3) are p, pi+, pi-
	RecoDecay::getDaughters(particlesMC, particle.globalIndex(), &arrDaughLcIndex, arrDaughLcPDGRef, 3); // best would be to check the K0S daughters
	LOG(INFO) << "arrDaughLcIndex.size() = " << arrDaughLcIndex.size();
	if (arrDaughLcIndex.size() == 3) {
	  for (auto iProng = 0; iProng < arrDaughLcIndex.size(); ++iProng) {
	    auto daughI = particlesMC.iteratorAt(arrDaughLcIndex[iProng]);
	    arrDaughLcPDG[iProng] = daughI.pdgCode();
	  }
	  if (!(arrDaughLcPDG[0] == arrDaughLcPDGRef[0] && arrDaughLcPDG[1] == arrDaughLcPDGRef[1] && arrDaughLcPDG[2] == arrDaughLcPDGRef[2])) { // this should be the condition, first bach, then v0
	    LOG(INFO) << "Pity, the three final daughters are not p, pi+, pi-, but " << arrDaughLcPDG[0] << ", " << arrDaughLcPDG[1] << ", " << arrDaughLcPDG[2];
	    sign = 0;
	  }
	  else {
	    LOG(INFO) << " YU-HUUUUU!";
	  }
	}
	//RecoDecay::isMatchedMCGen(particlesMC, particle, 4122, array{+kProton, +kPiPlus, -kPiPlus}, true, &sign, 3);
	//if (sign !=0) {
	// LOG(INFO) << "Lc in p pi+ pi-";
	//}
      }
      rowMCMatchGen(sign);
    }
  }
};

//____________________________________________________________________

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<HFCandidateCreatorCascade>("hf-cand-creator-cascade"),
    adaptAnalysisTask<HFCandidateCreatorCascadeExpressions>("hf-cand-creator-cascade-expressions")};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<HFCandidateCreatorCascadeMC>("hf-cand-creator-cascade-mc"));
  }
  return workflow;
}
