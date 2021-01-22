// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFCandidateCreator3Prong.cxx
/// \brief Reconstruction of heavy-flavour 3-prong decay candidates
/// \note Extended from HFCandidateCreator2Prong
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Perform MC matching."}};
  workflowOptions.push_back(optionDoMC);
  ConfigParamSpec optionWriteTree{"writeTree", VariantType::Bool, false, {"Writing debug tree."}};
  workflowOptions.push_back(optionWriteTree);
}

#include "Framework/runDataProcessing.h"

/// Reconstruction of heavy-flavour 3-prong decay candidates
struct HFCandidateCreator3Prong {
  Produces<aod::HfCandProng3Base> rowCandidateBase;

  Configurable<double> magneticField{"d_bz", 5., "magnetic field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations is chi2/chi2old > this"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};

  OutputObj<TH1F> hmass3{TH1F("hmass3", "3-prong candidates;inv. mass (#pi K #pi) (GeV/#it{c}^{2});entries", 500, 1.6, 2.1)};
  OutputObj<TH1F> hCovPVXX{TH1F("hCovPVXX", "3-prong candidates;XX element of cov. matrix of prim. vtx position (cm^{2});entries", 100, 0., 1.e-4)};
  OutputObj<TH1F> hCovSVXX{TH1F("hCovSVXX", "3-prong candidates;XX element of cov. matrix of sec. vtx position (cm^{2});entries", 100, 0., 0.2)};

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double massPiKPi{0.};

  void process(aod::Collisions const& collisions,
               aod::HfTrackIndexProng3 const& rowsTrackIndexProng3,
               aod::BigTracks const& tracks)
  {
    // 3-prong vertex fitter
    o2::vertexing::DCAFitterN<3> df;
    df.setBz(magneticField);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);
    df.setUseAbsDCA(true);

    // loop over pairs of track indeces
    for (const auto& rowTrackIndexProng3 : rowsTrackIndexProng3) {
      auto trackParVar0 = getTrackParCov(rowTrackIndexProng3.index0());
      auto trackParVar1 = getTrackParCov(rowTrackIndexProng3.index1());
      auto trackParVar2 = getTrackParCov(rowTrackIndexProng3.index2());
      auto collision = rowTrackIndexProng3.index0().collision();

      // reconstruct the 3-prong secondary vertex
      if (df.process(trackParVar0, trackParVar1, trackParVar2) == 0) {
        continue;
      }
      const auto& secondaryVertex = df.getPCACandidate();
      auto chi2PCA = df.getChi2AtPCACandidate();
      auto covMatrixPCA = df.calcPCACovMatrix().Array();
      hCovSVXX->Fill(covMatrixPCA[0]); // FIXME: Calculation of errorDecayLength(XY) gives wrong values without this line.
      trackParVar0 = df.getTrack(0);
      trackParVar1 = df.getTrack(1);
      trackParVar2 = df.getTrack(2);

      // get track momenta
      array<float, 3> pvec0;
      array<float, 3> pvec1;
      array<float, 3> pvec2;
      trackParVar0.getPxPyPzGlo(pvec0);
      trackParVar1.getPxPyPzGlo(pvec1);
      trackParVar2.getPxPyPzGlo(pvec2);

      // get track impact parameters
      // This modifies track momenta!
      auto primaryVertex = getPrimaryVertex(collision);
      auto covMatrixPV = primaryVertex.getCov();
      hCovPVXX->Fill(covMatrixPV[0]);
      o2::dataformats::DCA impactParameter0;
      o2::dataformats::DCA impactParameter1;
      o2::dataformats::DCA impactParameter2;
      trackParVar0.propagateToDCA(primaryVertex, magneticField, &impactParameter0);
      trackParVar1.propagateToDCA(primaryVertex, magneticField, &impactParameter1);
      trackParVar2.propagateToDCA(primaryVertex, magneticField, &impactParameter2);

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
                       pvec0[0], pvec0[1], pvec0[2],
                       pvec1[0], pvec1[1], pvec1[2],
                       pvec2[0], pvec2[1], pvec2[2],
                       impactParameter0.getY(), impactParameter1.getY(), impactParameter2.getY(),
                       std::sqrt(impactParameter0.getSigmaY2()), std::sqrt(impactParameter1.getSigmaY2()), std::sqrt(impactParameter2.getSigmaY2()),
                       rowTrackIndexProng3.index0Id(), rowTrackIndexProng3.index1Id(), rowTrackIndexProng3.index2Id(),
                       rowTrackIndexProng3.hfflag());

      // fill histograms
      if (b_dovalplots) {
        // calculate invariant mass
        auto arrayMomenta = array{pvec0, pvec1, pvec2};
        massPiKPi = RecoDecay::M(std::move(arrayMomenta), array{massPi, massK, massPi});
        hmass3->Fill(massPiKPi);
      }
    }
  }
};

/// Extends the base table with expression columns.
struct HFCandidateCreator3ProngExpressions {
  Spawns<aod::HfCandProng3Ext> rowCandidateProng3;
  void init(InitContext const&) {}
};

/// Performs MC matching.
struct HFCandidateCreator3ProngMC {
  Produces<aod::HfCandProng3MCRec> rowMCMatchRec;
  Produces<aod::HfCandProng3MCGen> rowMCMatchGen;

  void process(aod::HfCandProng3 const& candidates,
               aod::BigTracksMC const& tracks,
               aod::McParticles const& particlesMC)
  {
    int8_t sign = 0;
    int8_t result = N3ProngDecays;

    // Match reconstructed candidates.
    for (auto& candidate : candidates) {
      //Printf("New rec. candidate");
      result = N3ProngDecays;
      auto arrayDaughters = array{candidate.index0_as<aod::BigTracksMC>(), candidate.index1_as<aod::BigTracksMC>(), candidate.index2_as<aod::BigTracksMC>()};

      // D± → π± K∓ π±
      //Printf("Checking D± → π± K∓ π±");
      if (RecoDecay::getMatchedMCRec(particlesMC, arrayDaughters, 411, array{+kPiPlus, -kKPlus, +kPiPlus}, true, &sign) > -1) {
        result = sign * DPlusToPiKPi;
      }

      // Λc± → p± K∓ π±
      if (result == N3ProngDecays) {
        //Printf("Checking Λc± → p± K∓ π±");
        if (RecoDecay::getMatchedMCRec(particlesMC, std::move(arrayDaughters), 4122, array{+kProton, -kKPlus, +kPiPlus}, true, &sign) > -1) {
          result = sign * LcToPKPi;
        }
      }

      rowMCMatchRec(result);
    }

    // Match generated particles.
    for (auto& particle : particlesMC) {
      //Printf("New gen. candidate");
      result = N3ProngDecays;

      // D± → π± K∓ π±
      //Printf("Checking D± → π± K∓ π±");
      if (RecoDecay::isMatchedMCGen(particlesMC, particle, 411, array{+kPiPlus, -kKPlus, +kPiPlus}, true, &sign)) {
        result = sign * DPlusToPiKPi;
      }

      // Λc± → p± K∓ π±
      if (result == N3ProngDecays) {
        //Printf("Checking Λc± → p± K∓ π±");
        if (RecoDecay::isMatchedMCGen(particlesMC, particle, 4122, array{+kProton, -kKPlus, +kPiPlus}, true, &sign)) {
          result = sign * LcToPKPi;
        }
      }

      rowMCMatchGen(result);
    }
  }
};

namespace o2::aod
{
namespace full
{
DECLARE_SOA_COLUMN(XSecondaryVertex, xSecondaryVertex, float);
DECLARE_SOA_COLUMN(RSecondaryVertex, rSecondaryVertex, float);
DECLARE_SOA_COLUMN(PtProng0, ptProng0, float);
DECLARE_SOA_COLUMN(PProng0, pProng0, float);
DECLARE_SOA_COLUMN(ImpactParameterNormalised0, impactParameterNormalised0, float);
DECLARE_SOA_COLUMN(PtProng1, ptProng1, float);
DECLARE_SOA_COLUMN(PProng1, pProng1, float);
DECLARE_SOA_COLUMN(ImpactParameterNormalised1, impactParameterNormalised1, float);
DECLARE_SOA_COLUMN(PtProng2, ptProng2, float);
DECLARE_SOA_COLUMN(PProng2, pProng2, float);
DECLARE_SOA_COLUMN(ImpactParameterNormalised2, impactParameterNormalised2, float);
DECLARE_SOA_COLUMN(M, m, float);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(P, p, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(E, e, float);
DECLARE_SOA_COLUMN(NSigTPC_Pi_0, nsigTPC_Pi_0, float);
DECLARE_SOA_COLUMN(NSigTPC_Ka_0, nsigTPC_Ka_0, float);
DECLARE_SOA_COLUMN(NSigTOF_Pi_0, nsigTOF_Pi_0, float);
DECLARE_SOA_COLUMN(NSigTOF_Ka_0, nsigTOF_Ka_0, float);
DECLARE_SOA_COLUMN(NSigTPC_Pi_1, nsigTPC_Pi_1, float);
DECLARE_SOA_COLUMN(NSigTPC_Ka_1, nsigTPC_Ka_1, float);
DECLARE_SOA_COLUMN(NSigTOF_Pi_1, nsigTOF_Pi_1, float);
DECLARE_SOA_COLUMN(NSigTOF_Ka_1, nsigTOF_Ka_1, float);
DECLARE_SOA_COLUMN(NSigTPC_Pi_2, nsigTPC_Pi_2, float);
DECLARE_SOA_COLUMN(NSigTPC_Ka_2, nsigTPC_Ka_2, float);
DECLARE_SOA_COLUMN(NSigTOF_Pi_2, nsigTOF_Pi_2, float);
DECLARE_SOA_COLUMN(NSigTOF_Ka_2, nsigTOF_Ka_2, float);
DECLARE_SOA_COLUMN(DecayLength, decayLength, float);
DECLARE_SOA_COLUMN(DecayLengthXY, decayLengthXY, float);
DECLARE_SOA_COLUMN(DecayLengthNormalised, decayLengthNormalised, float);
DECLARE_SOA_COLUMN(DecayLengthXYNormalised, decayLengthXYNormalised, float);
DECLARE_SOA_COLUMN(CPA, cpa, float);
DECLARE_SOA_COLUMN(CPAXY, cpaXY, float);
DECLARE_SOA_COLUMN(Ct, ct, float);
DECLARE_SOA_COLUMN(MCflag, mcflag, uint8_t);
// Events
DECLARE_SOA_COLUMN(IsEventReject, isEventReject, int);
DECLARE_SOA_COLUMN(RunNumber, runNumber, int);
} // namespace full

DECLARE_SOA_TABLE(HfCandProng3Full, "AOD", "HFCANDP3Full",
                  collision::BCId,
                  collision::NumContrib,
                  collision::PosX,
                  collision::PosY,
                  collision::PosZ,
                  hf_cand::XSecondaryVertex,
                  hf_cand::YSecondaryVertex,
                  hf_cand::ZSecondaryVertex,
                  hf_cand::ErrorDecayLength,
                  hf_cand::ErrorDecayLengthXY,
                  hf_cand::Chi2PCA,
                  full::RSecondaryVertex,
                  full::DecayLength,
                  full::DecayLengthXY,
                  full::DecayLengthNormalised,
                  full::DecayLengthXYNormalised,
                  full::ImpactParameterNormalised0,
                  full::PtProng0,
                  full::PProng0,
                  full::ImpactParameterNormalised1,
                  full::PtProng1,
                  full::PProng1,
                  full::ImpactParameterNormalised2,
                  full::PtProng2,
                  full::PProng2,
                  hf_cand::PxProng0,
                  hf_cand::PyProng0,
                  hf_cand::PzProng0,
                  hf_cand::PxProng1,
                  hf_cand::PyProng1,
                  hf_cand::PzProng1,
                  hf_cand::PxProng2,
                  hf_cand::PyProng2,
                  hf_cand::PzProng2,
                  hf_cand::ImpactParameter0,
                  hf_cand::ImpactParameter1,
                  hf_cand::ImpactParameter2,
                  hf_cand::ErrorImpactParameter0,
                  hf_cand::ErrorImpactParameter1,
                  hf_cand::ErrorImpactParameter2,
                  full::NSigTPC_Pi_0,
                  full::NSigTPC_Ka_0,
                  full::NSigTOF_Pi_0,
                  full::NSigTOF_Ka_0,
                  full::NSigTPC_Pi_1,
                  full::NSigTPC_Ka_1,
                  full::NSigTOF_Pi_1,
                  full::NSigTOF_Ka_1,
                  full::NSigTPC_Pi_2,
                  full::NSigTPC_Ka_2,
                  full::NSigTOF_Pi_2,
                  full::NSigTOF_Ka_2,
                  full::M,
                  full::Pt,
                  full::P,
                  full::CPA,
                  full::CPAXY,
                  full::Ct,
                  full::Eta,
                  full::Phi,
                  full::Y,
                  full::E,
                  full::MCflag);

DECLARE_SOA_TABLE(HfCandProng3FullEvents, "AOD", "HFCANDP3FullE",
                  collision::BCId,
                  collision::NumContrib,
                  collision::PosX,
                  collision::PosY,
                  collision::PosZ,
                  full::IsEventReject,
                  full::RunNumber);

DECLARE_SOA_TABLE(HfCandProng3FullParticles, "AOD", "HFCANDP3FullP",
                  collision::BCId,
                  full::Pt,
                  full::Eta,
                  full::Phi,
                  full::Y,
                  full::MCflag);

} // namespace o2::aod

/// Writes a debug tree
struct CandidateTreeWriter {
  Produces<o2::aod::HfCandProng3Full> rowCandidateFull;
  Produces<o2::aod::HfCandProng3FullEvents> rowCandidateFullEvents;
  Produces<o2::aod::HfCandProng3FullParticles> rowCandidateFullParticles;
  void init(InitContext const&)
  {
  }
  void process(aod::Collisions const& collisions,
               aod::McCollisions const& mccollisions,
               soa::Join<aod::HfCandProng3, aod::HfCandProng3MCRec> const& candidates,
               soa::Join<aod::McParticles, aod::HfCandProng3MCGen> const& particles,
               aod::BigTracksPID const& tracks)
  {
    rowCandidateFullEvents.reserve(collisions.size());
    for (auto& collision : collisions) {
      rowCandidateFullEvents(
        collision.bcId(),
        collision.numContrib(),
        collision.posX(),
        collision.posY(),
        collision.posZ(),
        0,
        1);
    }
    rowCandidateFull.reserve(candidates.size());
    for (auto& candidate : candidates) {
      rowCandidateFull(
        candidate.index0_as<aod::BigTracksPID>().collision().bcId(),
        candidate.index0_as<aod::BigTracksPID>().collision().numContrib(),
        candidate.posX(),
        candidate.posY(),
        candidate.posZ(),
        candidate.xSecondaryVertex(),
        candidate.ySecondaryVertex(),
        candidate.zSecondaryVertex(),
        candidate.errorDecayLength(),
        candidate.errorDecayLengthXY(),
        candidate.chi2PCA(),
        candidate.rSecondaryVertex(),
        candidate.decayLength(),
        candidate.decayLengthXY(),
        candidate.decayLengthNormalised(),
        candidate.decayLengthXYNormalised(),
        candidate.impactParameterNormalised0(),
        TMath::Abs(candidate.ptProng0()),
        TMath::Sqrt(RecoDecay::P(candidate.pxProng0(), candidate.pyProng0(), candidate.pzProng0())),
        candidate.impactParameterNormalised1(),
        TMath::Abs(candidate.ptProng1()),
        TMath::Sqrt(RecoDecay::P(candidate.pxProng1(), candidate.pyProng1(), candidate.pzProng1())),
        candidate.impactParameterNormalised2(),
        TMath::Abs(candidate.ptProng2()),
        TMath::Sqrt(RecoDecay::P(candidate.pxProng2(), candidate.pyProng2(), candidate.pzProng2())),
        candidate.pxProng0(),
        candidate.pyProng0(),
        candidate.pzProng0(),
        candidate.pxProng1(),
        candidate.pyProng1(),
        candidate.pzProng1(),
        candidate.pxProng2(),
        candidate.pyProng2(),
        candidate.pzProng2(),
        candidate.impactParameter0(),
        candidate.impactParameter1(),
        candidate.impactParameter2(),
        candidate.errorImpactParameter0(),
        candidate.errorImpactParameter1(),
        candidate.errorImpactParameter2(),
        candidate.index0_as<aod::BigTracksPID>().tpcNSigmaPi(),
        candidate.index0_as<aod::BigTracksPID>().tpcNSigmaKa(),
        candidate.index0_as<aod::BigTracksPID>().tofNSigmaPi(),
        candidate.index0_as<aod::BigTracksPID>().tofNSigmaKa(),
        candidate.index1_as<aod::BigTracksPID>().tpcNSigmaPi(),
        candidate.index1_as<aod::BigTracksPID>().tpcNSigmaKa(),
        candidate.index1_as<aod::BigTracksPID>().tofNSigmaPi(),
        candidate.index1_as<aod::BigTracksPID>().tofNSigmaKa(),
        candidate.index2_as<aod::BigTracksPID>().tpcNSigmaPi(),
        candidate.index2_as<aod::BigTracksPID>().tpcNSigmaKa(),
        candidate.index2_as<aod::BigTracksPID>().tofNSigmaPi(),
        candidate.index2_as<aod::BigTracksPID>().tofNSigmaKa(),
        InvMassLcpKpi(candidate),
        candidate.pt(),
        candidate.p(),
        candidate.cpa(),
        candidate.cpaXY(),
        CtLc(candidate),
        candidate.eta(),
        candidate.phi(),
        YLc(candidate),
        ELc(candidate),
        candidate.flagMCMatchRec());
    }

    int npart = 0;
    rowCandidateFullParticles.reserve(particles.size());
    for (auto& particle : particles) {
      if (particle.flagMCMatchGen()) {
        rowCandidateFullParticles(
          particle.mcCollision().bcId(),
          particle.pt(),
          particle.eta(),
          particle.phi(),
          RecoDecay::Y(array{particle.px(), particle.py(), particle.pz()}, RecoDecay::getMassPDG(particle.pdgCode())),
          particle.flagMCMatchGen());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<HFCandidateCreator3Prong>("hf-cand-creator-3prong"),
    adaptAnalysisTask<HFCandidateCreator3ProngExpressions>("hf-cand-creator-3prong-expressions")};
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<HFCandidateCreator3ProngMC>("hf-cand-creator-3prong-mc"));
  }
  const bool writeTree = cfgc.options().get<bool>("writeTree");
  if (writeTree) {
    workflow.push_back(adaptAnalysisTask<CandidateTreeWriter>("hf-cand-tree-3prong-writer"));
  }
  return workflow;
}
