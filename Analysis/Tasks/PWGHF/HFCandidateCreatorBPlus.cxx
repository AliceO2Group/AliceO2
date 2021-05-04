// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFCandidateCreator2Prong.cxx
/// \brief Reconstruction of heavy-flavour 2-prong decay candidates
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN
/// \author Deepa Thomas <deepa.thomas@cern.ch>, UT Austin
/// \author Antonio Palasciano <antonio.palasciano@cern.ch>, Università degli Studi di Bari & INFN, Sezione di Bari

#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"
#include "ReconstructionDataFormats/V0.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::aod::hf_cand_bplus;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Perform MC matching."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

/// Reconstruction of B± → D0bar(D0) π± → (K± π∓) π±
struct HFCandidateCreatorBPlus {
  Produces<aod::HfCandBPlusBase> rowCandidateBase;

  // vertexing parameters
  Configurable<double> d_bz{"d_bz", 5., "magnetic field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations if chi2/chi2old > this"};
  Configurable<bool> d_UseAbsDCA{"d_UseAbsDCA", true, "Use Abs DCAs"};

  OutputObj<TH1F> hCovPVXX{TH1F("hCovPVXX", "2-prong candidates;XX element of cov. matrix of prim. vtx. position (cm^{2});entries", 100, 0., 1.e-4)};
  OutputObj<TH1F> hCovSVXX{TH1F("hCovSVXX", "2-prong candidates;XX element of cov. matrix of sec. vtx. position (cm^{2});entries", 100, 0., 0.2)};

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double massPiK{0.};
  double massKPi{0.};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", 1., "max. cand. pseudorapidity"};
  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(aod::Collision const& collisions,
               soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates,
               aod::BigTracks const& tracks)
  {
    //Initialise fitter for B vertex
    o2::vertexing::DCAFitterN<2> bfitter;
    bfitter.setBz(d_bz);
    bfitter.setPropagateToPCA(b_propdca);
    bfitter.setMaxR(d_maxr);
    bfitter.setMinParamChange(d_minparamchange);
    bfitter.setMinRelChi2Change(d_minrelchi2change);
    bfitter.setUseAbsDCA(d_UseAbsDCA);

    //Initial fitter to redo D-vertex to get extrapolated daughter tracks
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(d_bz);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);
    df.setUseAbsDCA(d_UseAbsDCA);

    // loop over pairs of track indices
    for (auto& candidate : candidates) {
      if (!(candidate.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (std::abs(candidate.eta()) > cutEtaCandMax) {
        continue;
      }
      if ((candidate.isSelD0bar() < d_selectionFlagD0bar) && (candidate.isSelD0() < d_selectionFlagD0))
        continue;

      double D0InvMass = -1;
      if (candidate.isSelD0bar() >= d_selectionFlagD0bar) {
        D0InvMass = InvMassD0bar(candidate);
      }
      if (candidate.isSelD0() >= d_selectionFlagD0) {
        D0InvMass = InvMassD0(candidate);
      }

      const std::array<float, 3> vertexD0 = {candidate.xSecondaryVertex(), candidate.ySecondaryVertex(), candidate.zSecondaryVertex()};
      const std::array<float, 3> momentumD0 = {candidate.px(), candidate.py(), candidate.pz()};

      auto prong0 = candidate.index0_as<aod::BigTracks>();
      auto prong1 = candidate.index1_as<aod::BigTracks>();
      auto prong0TrackParCov = getTrackParCov(prong0);
      auto prong1TrackParCov = getTrackParCov(prong1);
      auto collision = prong0.collision();

      // LOGF(INFO, "All track: %d (prong0); %d (prong1)", candidate.index0().globalIndex(), candidate.index1().globalIndex());
      // LOGF(INFO, "All track pT: %f (prong0); %f (prong1)", prong0.pt(), prong1.pt());

      // reconstruct D0 secondary vertex
      if (df.process(prong0TrackParCov, prong1TrackParCov) == 0) {
        continue;
      }

      prong0TrackParCov.propagateTo(candidate.xSecondaryVertex(), d_bz);
      prong1TrackParCov.propagateTo(candidate.xSecondaryVertex(), d_bz);
      const std::array<float, 6> D0CovMatrix = df.calcPCACovMatrixFlat();
      // build a D0 neutral track
      auto trackD0 = o2::dataformats::V0(vertexD0, momentumD0, D0CovMatrix, prong0TrackParCov, prong1TrackParCov, {0, 0}, {0, 0});

      //loop over tracks for pi selection
      auto count = 0;
      for (auto& track : tracks) {
        //if (count % 100 == 0) {
        //  LOGF(INFO, "Col: %d (cand); %d (track)", candidate.collisionId(), track.collisionId());
        //  count++;
        // }

        if (candidate.isSelD0() >= d_selectionFlagD0 && track.signed1Pt() > 0)
          continue; //to select D0pi- pair
        if (candidate.isSelD0bar() >= d_selectionFlagD0bar && track.signed1Pt() < 0)
          continue; //to select D0(bar)pi+ pair
        if (candidate.index0Id() == track.globalIndex() || candidate.index1Id() == track.globalIndex())
          continue; //daughter track id and bachelor track id not the same

        auto bachTrack = getTrackParCov(track);

        std::array<float, 3> pvecD0 = {0., 0., 0.};
        std::array<float, 3> pvecbach = {0., 0., 0.};
        std::array<float, 3> pvecBCand = {0., 0., 0.};

        //find the DCA between the D0 and the bachelor track, for B+
        int nCand = bfitter.process(trackD0, bachTrack); //Plot nCand

        if (nCand == 0)
          continue;

        bfitter.propagateTracksToVertex();          // propagate the bachelor and D0 to the B+ vertex
        bfitter.getTrack(0).getPxPyPzGlo(pvecD0);   //momentum of D0 at the B+ vertex
        bfitter.getTrack(1).getPxPyPzGlo(pvecbach); //momentum of pi+ at the B+ vertex
        const auto& BSecVertex = bfitter.getPCACandidate();
        auto chi2PCA = bfitter.getChi2AtPCACandidate();
        auto covMatrixPCA = bfitter.calcPCACovMatrix().Array();
        hCovSVXX->Fill(covMatrixPCA[0]); // FIXME: Calculation of errorDecayLength(XY) gives wrong values without this line.

        pvecBCand = array{pvecbach[0] + pvecD0[0],
                          pvecbach[1] + pvecD0[1],
                          pvecbach[2] + pvecD0[2]};

        // get track impact parameters
        // This modifies track momenta!
        auto primaryVertex = getPrimaryVertex(collision);
        auto covMatrixPV = primaryVertex.getCov();
        hCovPVXX->Fill(covMatrixPV[0]);
        o2::dataformats::DCA impactParameter0;
        o2::dataformats::DCA impactParameter1;
        bfitter.getTrack(0).propagateToDCA(primaryVertex, d_bz, &impactParameter0);
        bfitter.getTrack(1).propagateToDCA(primaryVertex, d_bz, &impactParameter1);

        // get uncertainty of the decay length
        double phi, theta;
        getPointDirection(array{collision.posX(), collision.posY(), collision.posZ()}, BSecVertex, phi, theta);
        auto errorDecayLength = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, theta) + getRotatedCovMatrixXX(covMatrixPCA, phi, theta));
        auto errorDecayLengthXY = std::sqrt(getRotatedCovMatrixXX(covMatrixPV, phi, 0.) + getRotatedCovMatrixXX(covMatrixPCA, phi, 0.));

        int hfFlag = 1 << BPlusToD0Pi;

        // fill candidate table rows
        rowCandidateBase(collision.globalIndex(),
                         collision.posX(), collision.posY(), collision.posZ(),
                         BSecVertex[0], BSecVertex[1], BSecVertex[2],
                         errorDecayLength, errorDecayLengthXY,
                         chi2PCA,
                         pvecD0[0], pvecD0[1], pvecD0[2],
                         pvecbach[0], pvecbach[1], pvecbach[2],
                         impactParameter0.getY(), impactParameter1.getY(),
                         std::sqrt(impactParameter0.getSigmaY2()), std::sqrt(impactParameter1.getSigmaY2()),
                         candidate.globalIndex(), track.globalIndex(), //index D0 and bachelor
                         hfFlag);

      } //track loop
    }   //D0 cand loop
  }     //process
};      //struct

/// Extends the base table with expression columns.
struct HFCandidateCreatorBPlusExpressions {
  Spawns<aod::HfCandBPlusExt> rowCandidateBPlus;
  void init(InitContext const&) {}
};

/// Performs MC matching.
struct HFCandidateCreatorBPlusMC {
  Produces<aod::HfCandBPMCRec> rowMCMatchRec;
  Produces<aod::HfCandBPMCGen> rowMCMatchGen;

  void process(aod::HfCandBPlus const& candidates,
               aod::HfCandProng2,
               aod::BigTracksMC const& tracks,
               aod::McParticles const& particlesMC)
  {
    int indexRec = -1;
    int8_t sign = 0;
    int8_t flag = 0;

    // Match reconstructed candidates.
    for (auto& candidate : candidates) {
      //Printf("New rec. candidate");
      flag = 0;
      auto D0barTrack = candidate.index0();
      auto arrayDaughters = array{candidate.index1_as<aod::BigTracksMC>(), D0barTrack.index0_as<aod::BigTracksMC>(), D0barTrack.index1_as<aod::BigTracksMC>()};

      // B± → D0bar(D0) π± → (K± π∓) π±
      //Printf("Checking B± → D0(bar) π±");
      indexRec = RecoDecay::getMatchedMCRec(particlesMC, arrayDaughters, 521, array{321, -kPiPlus, +kPiPlus}, true, &sign, 2);
      if (indexRec > -1) {
        flag = sign * (1 << BPlusToD0Pi);
      }

      rowMCMatchRec(flag);
    }

    // Match generated particles.
    for (auto& particle : particlesMC) {
      //Printf("New gen. candidate");
      flag = 0;
      // B± → D0bar(D0) π± → (K± π∓) π±
      //Printf("Checking B± → D0(bar) π±");
      if (RecoDecay::isMatchedMCGen(particlesMC, particle, 521, array{-421, +kPiPlus}, true, &sign, 1)) {
        // D0(bar) → π± K∓
        //Printf("Checking D0(bar) → π± K∓");
        std::vector<int> arrayDaughterB;
        RecoDecay::getDaughters(particlesMC, particle, &arrayDaughterB, array{421}, 1); //it takes abs. of PDG codes
        auto D0candidateMC = particlesMC.iteratorAt(arrayDaughterB[0]);
        if (RecoDecay::isMatchedMCGen(particlesMC, D0candidateMC, 421, array{-321, +kPiPlus}, true, &sign)) {
          flag = (-1) * sign * (1 << BPlusToD0Pi);
        }
      }

      rowMCMatchGen(flag);
    } //B candidates
  }   // process
};    // struct

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<HFCandidateCreatorBPlus>(cfgc, TaskName{"hf-cand-creator-bplus"}),
    adaptAnalysisTask<HFCandidateCreatorBPlusExpressions>(cfgc, TaskName{"hf-cand-creator-bplus-expressions"})};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<HFCandidateCreatorBPlusMC>(cfgc, TaskName{"hf-cand-creator-bplus-mc"}));
  }
  return workflow;
}