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

  Configurable<double> d_bz{"d_bz", 5., "magnetic field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations is chi2/chi2old > this"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};

  OutputObj<TH1F> hmass2{TH1F("hmass2", "2-prong candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", 500, 0., 5.)};
  OutputObj<TH1F> hCovPVXX{TH1F("hCovPVXX", "2-prong candidates;XX element of cov. matrix of prim. vtx. position (cm^{2});entries", 100, 0., 1.e-4)};
  OutputObj<TH1F> hCovSVXX{TH1F("hCovSVXX", "2-prong candidates;XX element of cov. matrix of sec. vtx. position (cm^{2});entries", 100, 0., 0.2)};

  double massP = RecoDecay::getMassPDG(kProton);
  double massK0s = RecoDecay::getMassPDG(kK0Short);
  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massLc = RecoDecay::getMassPDG(4122);
  double mass2K0sP{0.};

  void process(aod::Collisions const& collisions,
               aod::HfTrackIndexCasc const& rowsTrackIndexCasc,
               aod::BigTracks const& tracks,
               aod::V0DataExt const& V0s)
  {
    // 2-prong vertex fitter
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(d_bz);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);
    df.setUseAbsDCA(true);

    // loop over pairs of track indeces
    for (const auto& casc : rowsTrackIndexCasc) {
      auto trackParCovBach = getTrackParCov(casc.index0());
      const auto& v0 = casc.indexV0_as<o2::aod::V0DataExt>();
      const auto& posTrack = v0.posTrack_as<aod::BigTracks>();
      const auto& negTrack = v0.negTrack_as<aod::BigTracks>();
      auto posTrackParCov = getTrackParCov(posTrack); // check that BigTracks does not need TracksExtended!
      auto negTrackParCov = getTrackParCov(negTrack); // check that BigTracks does not need TracksExtended!
      posTrackParCov.propagateTo(v0.posX(), d_bz);    // propagate the track to the X closest to the V0 vertex
      negTrackParCov.propagateTo(v0.negX(), d_bz);    // propagate the track to the X closest to the V0 vertex
      const std::array<float, 3> vertexV0 = {v0.x(), v0.y(), v0.z()};
      const std::array<float, 3> momentumV0 = {v0.px(), v0.py(), v0.pz()};
      // we build the neutral track to then build the cascade
      auto trackV0 = o2::dataformats::V0(vertexV0, momentumV0, posTrackParCov, negTrackParCov, posTrack.globalIndex(), negTrack.globalIndex()); // build the V0 track

      auto collision = casc.index0().collision();

      // reconstruct the cascade secondary vertex
      if (df.process(trackV0, trackParCovBach) == 0) {
        continue;
      }
      const auto& secondaryVertex = df.getPCACandidate();
      auto chi2PCA = df.getChi2AtPCACandidate();
      auto covMatrixPCA = df.calcPCACovMatrix().Array();
      hCovSVXX->Fill(covMatrixPCA[0]); // FIXME: Calculation of errorDecayLength(XY) gives wrong values without this line.
      // do I have to call "df.propagateTracksToVertex();"?
      auto trackParVarv0 = df.getTrack(0);
      auto trackParVarbach = df.getTrack(1);

      // get track momenta
      array<float, 3> pvecv0;
      array<float, 3> pvecbach;
      trackParVarv0.getPxPyPzGlo(pvecv0);
      trackParVarbach.getPxPyPzGlo(pvecbach);

      // get track impact parameters
      // This modifies track momenta!
      auto primaryVertex = getPrimaryVertex(collision);
      auto covMatrixPV = primaryVertex.getCov();
      hCovPVXX->Fill(covMatrixPV[0]);
      o2::dataformats::DCA impactParameterv0;
      o2::dataformats::DCA impactParameterbach;
      trackParVarv0.propagateToDCA(primaryVertex, d_bz, &impactParameterv0); // we do this wrt the primary vtx
      trackParVarbach.propagateToDCA(primaryVertex, d_bz, &impactParameterbach);

      // get uncertainty of the decay length
      double phi, theta;
      getPointDirection(array{collision.posX(), collision.posY(), collision.posZ()}, secondaryVertex, phi, theta);
      auto errorDecayLength = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, theta) + getRotatedCovMatrixXX(covMatrixPCA, phi, theta));
      auto errorDecayLengthXY = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, 0.) + getRotatedCovMatrixXX(covMatrixPCA, phi, 0.));

      // fill candidate table rows
      rowCandidateBase(collision.posX(), collision.posY(), collision.posZ(),
                       secondaryVertex[0], secondaryVertex[1], secondaryVertex[2],
                       errorDecayLength, errorDecayLengthXY,
                       chi2PCA,
                       pvecv0[0], pvecv0[1], pvecv0[2],
                       pvecbach[0], pvecbach[1], pvecbach[2],
                       impactParameterv0.getY(), impactParameterbach.getY(),
                       std::sqrt(impactParameterv0.getSigmaY2()), std::sqrt(impactParameterbach.getSigmaY2()),
                       casc.indexV0Id(), casc.index0Id(),
                       casc.hfflag(),
                       v0.x(), v0.y(), v0.z(),
                       v0.pxpos(), v0.pypos(), v0.pzpos(),
                       v0.pxneg(), v0.pyneg(), v0.pzneg(),
                       v0.dcaV0daughters(),
                       v0.dcapostopv(),
                       v0.dcanegtopv());

      //--> Missing: DCA between V0 daughters; radius V0; cosPA --> can these be dynamic? to be checked

      // fill histograms
      if (b_dovalplots) {
        // calculate invariant masses
        auto arrayMomenta = array{pvecv0, pvecbach};
        mass2K0sP = RecoDecay::M(arrayMomenta, array{massK0s, massP});
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
/*
/// Performs MC matching.
struct HFCandidateCreator2ProngMC {
  Produces<aod::HfCandProng2MCRec> rowMCMatchRec;
  Produces<aod::HfCandProng2MCGen> rowMCMatchGen;

  void process(aod::HfCandProng2 const& candidates,
               aod::BigTracksMC const& tracks,
               aod::McParticles const& particlesMC)
  {
    int8_t sign = 0;
    int8_t result = N2ProngDecays;

    // Match reconstructed candidates.
    for (auto& candidate : candidates) {
      //Printf("New rec. candidate");
      result = N2ProngDecays;
      auto arrayDaughters = array{candidate.index0_as<aod::BigTracksMC>(), candidate.index1_as<aod::BigTracksMC>()};

      // D0(bar) → π± K∓
      //Printf("Checking D0(bar) → π± K∓");
      if (RecoDecay::getMatchedMCRec(particlesMC, std::move(arrayDaughters), 421, array{+kPiPlus, -kKPlus}, true, &sign) > -1) {
        result = sign * D0ToPiK;
      }

      rowMCMatchRec(result);
    }

    // Match generated particles.
    for (auto& particle : particlesMC) {
      //Printf("New gen. candidate");
      result = N2ProngDecays;

      // D0(bar) → π± K∓
      //Printf("Checking D0(bar) → π± K∓");
      if (RecoDecay::isMatchedMCGen(particlesMC, particle, 421, array{+kPiPlus, -kKPlus}, true, &sign)) {
        result = sign * D0ToPiK;
      }

      rowMCMatchGen(result);
    }
  }
};
*/
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<HFCandidateCreatorCascade>("hf-cand-creator-cascade"),
    adaptAnalysisTask<HFCandidateCreatorCascadeExpressions>("hf-cand-creator-cascade-expressions")};
  const bool doMC = cfgc.options().get<bool>("doMC");
  //if (doMC) {
  //  workflow.push_back(adaptAnalysisTask<HFCandidateCreator2ProngMC>("hf-cand-creator-2prong-mc"));
  //}
  return workflow;
}
