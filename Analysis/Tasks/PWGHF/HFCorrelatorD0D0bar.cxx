// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFCorrelatorD0D0bar.cxx
/// \brief D0-D0bar correlator task - data-like, MC-reco and MC-kine analyses. For ULS and LS pairs
///
/// \author Fabio Colamaria <fabio.colamaria@ba.infn.it>, INFN Bari

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::aod::hf_correlation_ddbar;
using namespace o2::analysis::hf_cuts_d0_topik;
using namespace o2::constants::math;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoLikeSign{"doLikeSign", VariantType::Bool, false, {"Run Like-Sign analysis."}};
  ConfigParamSpec optionDoMCccbar{"doMCccbar", VariantType::Bool, false, {"Run MC-Gen dedicated tasks."}};
  ConfigParamSpec optionDoMCGen{"doMCGen", VariantType::Bool, false, {"Run MC-Gen dedicated tasks."}};
  ConfigParamSpec optionDoMCRec{"doMCRec", VariantType::Bool, false, {"Run MC-Rec dedicated tasks."}};
  workflowOptions.push_back(optionDoLikeSign);
  workflowOptions.push_back(optionDoMCccbar);
  workflowOptions.push_back(optionDoMCGen);
  workflowOptions.push_back(optionDoMCRec);
}

#include "Framework/runDataProcessing.h"

///
/// Returns deltaPhi value in range [-pi/2., 3.*pi/2], typically used for correlation studies
///
double getDeltaPhi(double phiD, double phiDbar)
{
  return RecoDecay::constrainAngle(phiDbar - phiD, -o2::constants::math::PI / 2.);
}

/// definition of variables for D0D0bar pairs vs eta acceptance studies (hDDbarVsEtaCut, in data-like, MC-reco and MC-kine tasks)
const double maxEtaCut = 5.;
const double ptThresholdForMaxEtaCut = 10.;
const double incrementEtaCut = 0.1;
const double incrementPtThreshold = 0.5;
const double epsilon = 1E-5;

/// D0-D0bar correlation pair builder - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
struct HfCorrelatorD0D0bar {
  Produces<aod::DDbarPair> entryD0D0barPair;
  Produces<aod::DDbarRecoInfo> entryD0D0barRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCand", "D0,D0bar candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0", "D0,D0bar candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1", "D0,D0bar candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatus", "D0,D0bar candidates;selection status;entries", {HistType::kTH1F, {{4, -0.5, 3.5}}}},
     {"hEta", "D0,D0bar candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhi", "D0,D0bar candidates;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hY", "D0,D0bar candidates;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDDbarVsEtaCut", "D0,D0bar pairs vs #eta cut;#eta_{max};entries", {HistType::kTH2F, {{(int)(maxEtaCut / incrementEtaCut), 0., maxEtaCut}, {(int)(ptThresholdForMaxEtaCut / incrementPtThreshold), 0., ptThresholdForMaxEtaCut}}}}}};

  Configurable<int> selectionFlagD0{"selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> selectionFlagD0bar{"selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMass", "D0,D0bar candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0", "D0,D0bar candidates;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0bar", "D0,D0bar candidates;inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates)
  {
    for (auto& candidate1 : candidates) {
      if (cutYCandMax >= 0. && std::abs(YD0(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && candidate1.pt() < cutPtCandMin) {
        continue;
      }
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << DecayType::D0ToPiK)) {
        continue;
      }
      //fill invariant mass plots and generic info from all D0/D0bar candidates
      if (candidate1.isSelD0() >= selectionFlagD0) {
        registry.fill(HIST("hMass"), InvMassD0(candidate1), candidate1.pt());
        registry.fill(HIST("hMassD0"), InvMassD0(candidate1), candidate1.pt());
      }
      if (candidate1.isSelD0bar() >= selectionFlagD0bar) {
        registry.fill(HIST("hMass"), InvMassD0bar(candidate1), candidate1.pt());
        registry.fill(HIST("hMassD0bar"), InvMassD0bar(candidate1), candidate1.pt());
      }
      registry.fill(HIST("hPtCand"), candidate1.pt());
      registry.fill(HIST("hPtProng0"), candidate1.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate1.ptProng1());
      registry.fill(HIST("hEta"), candidate1.eta());
      registry.fill(HIST("hPhi"), candidate1.phi());
      registry.fill(HIST("hY"), YD0(candidate1));
      registry.fill(HIST("hSelectionStatus"), candidate1.isSelD0bar() + (candidate1.isSelD0() * 2));

      //D-Dbar correlation dedicated section
      //if the candidate is a D0, search for D0bar and evaluate correlations
      if (candidate1.isSelD0() < selectionFlagD0) {
        continue;
      }
      for (auto& candidate2 : candidates) {
        if (!(candidate2.hfflag() & 1 << DecayType::D0ToPiK)) { //check decay channel flag for candidate2
          continue;
        }
        if (candidate2.isSelD0bar() < selectionFlagD0bar) { //keep only D0bar candidates passing the selection
          continue;
        }
        //kinematic selection on D0bar candidates
        if (cutYCandMax >= 0. && std::abs(YD0(candidate2)) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && candidate2.pt() < cutPtCandMin) {
          continue;
        }
        //excluding trigger self-correlations (possible in case of both mass hypotheses accepted)
        if (candidate1.mRowIndex == candidate2.mRowIndex) {
          continue;
        }
        entryD0D0barPair(getDeltaPhi(candidate2.phi(), candidate1.phi()),
                         candidate2.eta() - candidate1.eta(),
                         candidate1.pt(),
                         candidate2.pt());
        entryD0D0barRecoInfo(InvMassD0(candidate1),
                             InvMassD0bar(candidate2),
                             0);
        double etaCut = 0.;
        double ptCut = 0.;
        do { //fill pairs vs etaCut plot
          ptCut = 0.;
          etaCut += incrementEtaCut;
          do { //fill pairs vs etaCut plot
            if (std::abs(candidate1.eta()) < etaCut && std::abs(candidate2.eta()) < etaCut && candidate1.pt() > ptCut && candidate2.pt() > ptCut) {
              registry.fill(HIST("hDDbarVsEtaCut"), etaCut - epsilon, ptCut + epsilon);
            }
            ptCut += incrementPtThreshold;
          } while (ptCut < ptThresholdForMaxEtaCut - epsilon);
        } while (etaCut < maxEtaCut - epsilon);
        //note: candidates selected as both D0 and D0bar are used, and considered in both situation (but not auto-correlated): reflections could play a relevant role.
        //another, more restrictive, option, could be to consider only candidates selected with a single option (D0 xor D0bar)

      } // end inner loop (Dbars)

    } //end outer loop
  }
};

/// D0-D0bar correlation pair builder - for MC reco-level analysis (candidates matched to true signal only, but also the various bkg sources are studied)
struct HfCorrelatorD0D0barMcRec {

  Produces<aod::DDbarPair> entryD0D0barPair;
  Produces<aod::DDbarRecoInfo> entryD0D0barRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCandMCRec", "D0,D0bar candidates - MC reco;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0MCRec", "D0,D0bar candidates - MC reco;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1MCRec", "D0,D0bar candidates - MC reco;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatusMCRec", "D0,D0bar candidates - MC reco;selection status;entries", {HistType::kTH1F, {{4, -0.5, 3.5}}}},
     {"hEtaMCRec", "D0,D0bar candidates - MC reco;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCRec", "D0,D0bar candidates - MC reco;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCRec", "D0,D0bar candidates - MC reco;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDDbarVsEtaCut", "D0,D0bar pairs vs #eta cut;#eta_{max};entries", {HistType::kTH2F, {{(int)(maxEtaCut / incrementEtaCut), 0., maxEtaCut}, {(int)(ptThresholdForMaxEtaCut / incrementPtThreshold), 0., ptThresholdForMaxEtaCut}}}}}};

  Configurable<int> selectionFlagD0{"selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> selectionFlagD0bar{"selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMassD0MCRec", "D0,D0bar candidates - MC reco;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0barMCRec", "D0,D0bar candidates - MC reco;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HfCandProng2MCRec>> const& candidates)
  {
    //MC reco level
    bool flagD0Signal = false;
    bool flagD0barSignal = false;
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << DecayType::D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && candidate1.pt() < cutPtCandMin) {
        continue;
      }
      if (std::abs(candidate1.flagMCMatchRec()) == 1 << DecayType::D0ToPiK) {
        //fill invariant mass plots and generic info from all D0/D0bar candidates
        if (candidate1.isSelD0() >= selectionFlagD0 && candidate1.flagMCMatchRec() == 1 << DecayType::D0ToPiK) { //only reco and matched as D0
          registry.fill(HIST("hMassD0MCRec"), InvMassD0(candidate1), candidate1.pt());
        }
        if (candidate1.isSelD0bar() >= selectionFlagD0bar && candidate1.flagMCMatchRec() == -(1 << DecayType::D0ToPiK)) { //only reco and matched as D0bar
          registry.fill(HIST("hMassD0barMCRec"), InvMassD0bar(candidate1), candidate1.pt());
        }
        registry.fill(HIST("hPtCandMCRec"), candidate1.pt());
        registry.fill(HIST("hPtProng0MCRec"), candidate1.ptProng0());
        registry.fill(HIST("hPtProng1MCRec"), candidate1.ptProng1());
        registry.fill(HIST("hEtaMCRec"), candidate1.eta());
        registry.fill(HIST("hPhiMCRec"), candidate1.phi());
        registry.fill(HIST("hYMCRec"), YD0(candidate1));
        registry.fill(HIST("hSelectionStatusMCRec"), candidate1.isSelD0bar() + (candidate1.isSelD0() * 2));
      }

      //D-Dbar correlation dedicated section
      //if the candidate is selected ad D0, search for D0bar and evaluate correlations
      if (candidate1.isSelD0() < selectionFlagD0) { //discard candidates not selected as D0 in outer loop
        continue;
      }
      flagD0Signal = candidate1.flagMCMatchRec() == 1 << DecayType::D0ToPiK; //flagD0Signal 'true' if candidate1 matched to D0 (particle)
      for (auto& candidate2 : candidates) {
        if (!(candidate2.hfflag() & 1 << DecayType::D0ToPiK)) { //check decay channel flag for candidate2
          continue;
        }
        if (candidate2.isSelD0bar() < selectionFlagD0bar) { //discard candidates not selected as D0bar in inner loop
          continue;
        }
        flagD0barSignal = candidate2.flagMCMatchRec() == -(1 << DecayType::D0ToPiK); //flagD0barSignal 'true' if candidate2 matched to D0 (particle)
        if (cutYCandMax >= 0. && std::abs(YD0(candidate2)) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && candidate2.pt() < cutPtCandMin) {
          continue;
        }
        //Excluding trigger self-correlations (possible in case of both mass hypotheses accepted)
        if (candidate1.mRowIndex == candidate2.mRowIndex) {
          continue;
        }
        //choice of options (D0/D0bar signal/bkg)
        int pairSignalStatus = 0; //0 = bkg/bkg, 1 = bkg/sig, 2 = sig/bkg, 3 = sig/sig
        if (flagD0Signal) {
          pairSignalStatus += 2;
        }
        if (flagD0barSignal) {
          pairSignalStatus += 1;
        }
        entryD0D0barPair(getDeltaPhi(candidate2.phi(), candidate1.phi()),
                         candidate2.eta() - candidate1.eta(),
                         candidate1.pt(),
                         candidate2.pt());
        entryD0D0barRecoInfo(InvMassD0(candidate1),
                             InvMassD0bar(candidate2),
                             pairSignalStatus);
        double etaCut = 0.;
        double ptCut = 0.;
        do { //fill pairs vs etaCut plot
          ptCut = 0.;
          etaCut += incrementEtaCut;
          do { //fill pairs vs etaCut plot
            if (std::abs(candidate1.eta()) < etaCut && std::abs(candidate2.eta()) < etaCut && candidate1.pt() > ptCut && candidate2.pt() > ptCut) {
              registry.fill(HIST("hDDbarVsEtaCut"), etaCut - epsilon, ptCut + epsilon);
            }
            ptCut += incrementPtThreshold;
          } while (ptCut < ptThresholdForMaxEtaCut - epsilon);
        } while (etaCut < maxEtaCut - epsilon);
      } // end inner loop (Dbars)

    } //end outer loop
  }
};

/// D0-D0bar correlation pair builder - for MC gen-level analysis (no filter/selection, only true signal)
struct HfCorrelatorD0D0barMcGen {

  Produces<aod::DDbarPair> entryD0D0barPair;

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hPtCandMCGen", "D0,D0bar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hEtaMCGen", "D0,D0bar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "D0,D0bar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCGen", "D0,D0bar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hcountD0D0barPerEvent", "D0,D0bar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hDDbarVsEtaCut", "D0,D0bar pairs vs #eta cut of D mesons;#eta_{max};entries", {HistType::kTH2F, {{(int)(maxEtaCut / incrementEtaCut), 0., maxEtaCut}, {(int)(ptThresholdForMaxEtaCut / incrementPtThreshold), 0., ptThresholdForMaxEtaCut}}}},
     {"hDDbarVsDaughterEtaCut", "D0,D0bar pairs vs #eta cut on D daughters;#eta_{max};entries", {HistType::kTH2F, {{(int)(maxEtaCut / incrementEtaCut), 0., maxEtaCut}, {(int)(ptThresholdForMaxEtaCut / incrementPtThreshold), 0., ptThresholdForMaxEtaCut}}}}}};

  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for trigger counters"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hcountD0triggersMCGen", "D0 trigger particles - MC gen;;N of trigger D0", {HistType::kTH2F, {{1, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    int counterD0D0bar = 0;
    registry.fill(HIST("hMCEvtCount"), 0);
    //MC gen level
    for (auto& particle1 : particlesMC) {
      //check if the particle is D0 or D0bar (for general plot filling and selection, so both cases are fine) - NOTE: decay channel is not probed!
      if (std::abs(particle1.pdgCode()) != pdg::Code::kD0) {
        continue;
      }
      double yD = RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()));
      if (cutYCandMax >= 0. && std::abs(yD) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && particle1.pt() < cutPtCandMin) {
        continue;
      }
      registry.fill(HIST("hPtCandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), yD);
      counterD0D0bar++;

      //D-Dbar correlation dedicated section
      //if it's a D0 particle, search for D0bar and evaluate correlations
      if (particle1.pdgCode() != pdg::Code::kD0) { //just checking the particle PDG, not the decay channel (differently from Reco: you have a BR factor btw such levels!)
        continue;
      }
      registry.fill(HIST("hcountD0triggersMCGen"), 0, particle1.pt()); //to count trigger D0 (for normalisation)
      for (auto& particle2 : particlesMC) {
        if (particle2.pdgCode() != pdg::Code::kD0bar) { //check that inner particle is D0bar
          continue;
        }
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && particle2.pt() < cutPtCandMin) {
          continue;
        }
        entryD0D0barPair(getDeltaPhi(particle2.phi(), particle1.phi()),
                         particle2.eta() - particle1.eta(),
                         particle1.pt(),
                         particle2.pt());
        double etaCut = 0.;
        double ptCut = 0.;

        //fill pairs vs etaCut plot
        bool rightDecayChannels = false;
        if ((std::abs(particle1.flagMCMatchGen()) == 1 << DecayType::D0ToPiK) && (std::abs(particle2.flagMCMatchGen()) == 1 << DecayType::D0ToPiK)) {
          rightDecayChannels = true;
        }
        do {
          ptCut = 0.;
          etaCut += incrementEtaCut;
          do {                                                                                                                                  //fill pairs vs etaCut plot
            if (std::abs(particle1.eta()) < etaCut && std::abs(particle2.eta()) < etaCut && particle1.pt() > ptCut && particle2.pt() > ptCut) { //fill with D and Dbar acceptance checks
              registry.fill(HIST("hDDbarVsEtaCut"), etaCut - epsilon, ptCut + epsilon);
            }
            if (rightDecayChannels) { //fill with D and Dbar daughter particls acceptance checks
              double etaCandidate1Daughter1 = particlesMC.iteratorAt(particle1.daughter0()).eta();
              double etaCandidate1Daughter2 = particlesMC.iteratorAt(particle1.daughter1()).eta();
              double etaCandidate2Daughter1 = particlesMC.iteratorAt(particle2.daughter0()).eta();
              double etaCandidate2Daughter2 = particlesMC.iteratorAt(particle2.daughter1()).eta();
              if (std::abs(etaCandidate1Daughter1) < etaCut && std::abs(etaCandidate1Daughter2) < etaCut &&
                  std::abs(etaCandidate2Daughter1) < etaCut && std::abs(etaCandidate2Daughter2) < etaCut &&
                  particle1.pt() > ptCut && particle2.pt() > ptCut) {
                registry.fill(HIST("hDDbarVsDaughterEtaCut"), etaCut - epsilon, ptCut + epsilon);
              }
            }
            ptCut += incrementPtThreshold;
          } while (ptCut < ptThresholdForMaxEtaCut - epsilon);
        } while (etaCut < maxEtaCut - epsilon);
      } //end inner loop
    }   //end outer loop
    registry.fill(HIST("hcountD0D0barPerEvent"), counterD0D0bar);
  }
};

/// D0-D0bar correlation pair builder - LIKE SIGN - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
/// NOTE: At the moment, both dPhi-symmetrical correlation pairs (part1-part2 and part2-part1) are filled,
///       since we bin in pT and selecting as trigger the largest pT particle would bias the distributions w.r.t. the ULS case.
struct HfCorrelatorD0D0barLs {

  Produces<aod::DDbarPair> entryD0D0barPair;
  Produces<aod::DDbarRecoInfo> entryD0D0barRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCand", "D0,D0bar candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0", "D0,D0bar candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1", "D0,D0bar candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatus", "D0,D0bar candidates;selection status;entries", {HistType::kTH1F, {{4, -0.5, 3.5}}}},
     {"hEta", "D0,D0bar candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhi", "D0,D0bar candidates;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hY", "D0,D0bar candidates;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}}}};

  Configurable<int> selectionFlagD0{"selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> selectionFlagD0bar{"selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMass", "D0,D0bar candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0", "D0,D0bar candidates;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0bar", "D0,D0bar candidates;inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates)
  {
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << DecayType::D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && candidate1.pt() < cutPtCandMin) {
        continue;
      }
      //fill invariant mass plots and generic info from all D0/D0bar candidates
      if (candidate1.isSelD0() >= selectionFlagD0) {
        registry.fill(HIST("hMass"), InvMassD0(candidate1), candidate1.pt());
        registry.fill(HIST("hMassD0"), InvMassD0(candidate1), candidate1.pt());
      }
      if (candidate1.isSelD0bar() >= selectionFlagD0bar) {
        registry.fill(HIST("hMass"), InvMassD0bar(candidate1), candidate1.pt());
        registry.fill(HIST("hMassD0bar"), InvMassD0bar(candidate1), candidate1.pt());
      }
      registry.fill(HIST("hPtCand"), candidate1.pt());
      registry.fill(HIST("hPtProng0"), candidate1.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate1.ptProng1());
      registry.fill(HIST("hEta"), candidate1.eta());
      registry.fill(HIST("hPhi"), candidate1.phi());
      registry.fill(HIST("hY"), YD0(candidate1));
      registry.fill(HIST("hSelectionStatus"), candidate1.isSelD0bar() + (candidate1.isSelD0() * 2));

      //D-Dbar correlation dedicated section
      //For like-sign, first loop on both D0 and D0bars. First candidate is for sure a D0 and D0bars (checked before, so don't re-check anything on it)
      for (auto& candidate2 : candidates) {
        //check decay channel flag for candidate2
        if (!(candidate2.hfflag() & 1 << DecayType::D0ToPiK)) {
          continue;
        }
        //for the associated, has to have smaller pT, and pass D0sel if trigger passes D0sel, or D0barsel if trigger passes D0barsel
        if ((candidate1.isSelD0() >= selectionFlagD0 && candidate2.isSelD0() >= selectionFlagD0) || (candidate1.isSelD0bar() >= selectionFlagD0bar && candidate2.isSelD0bar() >= selectionFlagD0bar)) {
          if (cutYCandMax >= 0. && std::abs(YD0(candidate2)) > cutYCandMax) {
            continue;
          }
          if (cutPtCandMin >= 0. && candidate2.pt() < cutPtCandMin) {
            continue;
          }
          //Excluding self-correlations
          if (candidate1.mRowIndex == candidate2.mRowIndex) {
            continue;
          }
          entryD0D0barPair(getDeltaPhi(candidate2.phi(), candidate1.phi()),
                           candidate2.eta() - candidate1.eta(),
                           candidate1.pt(),
                           candidate2.pt());
          entryD0D0barRecoInfo(InvMassD0(candidate1),
                               InvMassD0bar(candidate2),
                               0);
        }
        //note: candidates selected as both D0 and D0bar are used, and considered in both situation (but not auto-correlated): reflections could play a relevant role.
        //another, more restrictive, option, could be to consider only candidates selected with a single option (D0 xor D0bar)
      } // end inner loop (Dbars)
    }   //end outer loop
  }
};

/// D0-D0bar correlation pair builder - LIKE SIGN - for MC reco analysis (data-like but matching to true DO and D0bar)
/// NOTE: At the moment, both dPhi-symmetrical correlation pairs (part1-part2 and part2-part1) are filled,
///       since we bin in pT and selecting as trigger the largest pT particle would bias the distributions w.r.t. the ULS case.
struct HfCorrelatorD0D0barMcRecLs {

  Produces<aod::DDbarPair> entryD0D0barPair;
  Produces<aod::DDbarRecoInfo> entryD0D0barRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCandMCRec", "D0,D0bar candidates - MC reco;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0MCRec", "D0,D0bar candidates - MC reco;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1MCRec", "D0,D0bar candidates - MC reco;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatusMCRec", "D0,D0bar candidates - MC reco;selection status;entries", {HistType::kTH1F, {{4, -0.5, 3.5}}}},
     {"hEtaMCRec", "D0,D0bar candidates - MC reco;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCRec", "D0,D0bar candidates - MC reco;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCRec", "D0,D0bar candidates - MC reco;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}}}};

  Configurable<int> selectionFlagD0{"selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> selectionFlagD0bar{"selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMassD0MCRec", "D0,D0bar candidates - MC reco;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0barMCRec", "D0,D0bar candidates - MC reco;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HfCandProng2MCRec>> const& candidates)
  {
    //MC reco level
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << DecayType::D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && candidate1.pt() < cutPtCandMin) {
        continue;
      }
      if (std::abs(candidate1.flagMCMatchRec()) == 1 << DecayType::D0ToPiK) {
        //fill invariant mass plots and generic info from all D0/D0bar candidates
        if (candidate1.isSelD0() >= selectionFlagD0 && candidate1.flagMCMatchRec() == DecayType::D0ToPiK) { //only reco and matched as D0
          registry.fill(HIST("hMassD0MCRec"), InvMassD0(candidate1));
        }
        if (candidate1.isSelD0bar() >= selectionFlagD0bar && candidate1.flagMCMatchRec() == DecayType::D0ToPiK) { //only reco and matched as D0bar
          registry.fill(HIST("hMassD0barMCRec"), InvMassD0bar(candidate1));
        }
        registry.fill(HIST("hPtCandMCRec"), candidate1.pt());
        registry.fill(HIST("hPtProng0MCRec"), candidate1.ptProng0());
        registry.fill(HIST("hPtProng1MCRec"), candidate1.ptProng1());
        registry.fill(HIST("hEtaMCRec"), candidate1.eta());
        registry.fill(HIST("hPhiMCRec"), candidate1.phi());
        registry.fill(HIST("hYMCRec"), YD0(candidate1));
        registry.fill(HIST("hSelectionStatusMCRec"), candidate1.isSelD0bar() + (candidate1.isSelD0() * 2));

        //D-Dbar correlation dedicated section
        //For like-sign, first loop on both D0 and D0bars. First candidate is for sure a D0 and D0bars (looping on filtered) and was already matched, so don't re-check anything on it)
        for (auto& candidate2 : candidates) {
          //check decay channel flag for candidate2
          if (!(candidate2.hfflag() & 1 << DecayType::D0ToPiK)) {
            continue;
          }
          bool conditionLSForD0 = (candidate1.isSelD0() >= selectionFlagD0bar && candidate1.flagMCMatchRec() == 1 << DecayType::D0ToPiK) && (candidate2.isSelD0() >= selectionFlagD0bar && candidate2.flagMCMatchRec() == 1 << DecayType::D0ToPiK);
          bool conditionLSForD0bar = (candidate1.isSelD0bar() >= selectionFlagD0bar && candidate1.flagMCMatchRec() == -(1 << DecayType::D0ToPiK)) && (candidate2.isSelD0bar() >= selectionFlagD0bar && candidate2.flagMCMatchRec() == -(1 << DecayType::D0ToPiK));
          if (conditionLSForD0 || conditionLSForD0bar) { //LS pair (of D0 or of D0bar) + pt2<pt1
            if (cutYCandMax >= 0. && std::abs(YD0(candidate2)) > cutYCandMax) {
              continue;
            }
            if (cutPtCandMin >= 0. && candidate2.pt() < cutPtCandMin) {
              continue;
            }
            //Excluding self-correlations
            if (candidate1.mRowIndex == candidate2.mRowIndex) {
              continue;
            }
            entryD0D0barPair(getDeltaPhi(candidate2.phi(), candidate1.phi()),
                             candidate2.eta() - candidate1.eta(),
                             candidate1.pt(),
                             candidate2.pt());
            entryD0D0barRecoInfo(InvMassD0(candidate1),
                                 InvMassD0bar(candidate2),
                                 0); //for LS studies we set a dummy 0 for pairSignalStatus (there are no more the usual 4 possible combinations)

          } //end inner if (MC match)
        }   // end inner loop (Dbars)
      }     //end outer if (MC match)
    }       //end outer loop
  }
};

/// D0-D0bar correlation pair builder - for MC gen-level analysis, like sign particles
/// NOTE: At the moment, both dPhi-symmetrical correlation pairs (part1-part2 and part2-part1) are filled,
///       since we bin in pT and selecting as trigger the largest pT particle would bias the distributions w.r.t. the ULS case.
struct HfCorrelatorD0D0barMcGenLs {

  Produces<aod::DDbarPair> entryD0D0barPair;

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hPtCandMCGen", "D0,D0bar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hEtaMCGen", "D0,D0bar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "D0,D0bar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCGen", "D0,D0bar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hcountD0D0barPerEvent", "D0,D0bar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}}}};

  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for trigger counters"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hcountD0triggersMCGen", "D0 trigger particles - MC gen;;N of trigger D0", {HistType::kTH2F, {{1, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    int counterD0D0bar = 0;
    registry.fill(HIST("hMCEvtCount"), 0);
    //MC gen level
    for (auto& particle1 : particlesMC) {
      //check if the particle is D0 or D0bar (both can be trigger) - NOTE: decay channel is not probed!
      if (std::abs(particle1.pdgCode()) != pdg::Code::kD0) {
        continue;
      }
      double yD = RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()));
      if (cutYCandMax >= 0. && std::abs(yD) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && particle1.pt() < cutPtCandMin) {
        continue;
      }

      registry.fill(HIST("hPtCandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), yD);
      counterD0D0bar++;
      //D-Dbar correlation dedicated section
      //if it's D0, search for D0bar and evaluate correlations.
      registry.fill(HIST("hcountD0triggersMCGen"), 0, particle1.pt()); //to count trigger D0 (normalisation)
      for (auto& particle2 : particlesMC) {
        if (std::abs(particle2.pdgCode()) != pdg::Code::kD0) { //check that associated is a D0/D0bar (both are fine)
          continue;
        }
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && particle2.pt() < cutPtCandMin) {
          continue;
        }
        if (particle2.pdgCode() == particle1.pdgCode()) { //like-sign condition (both 421 or both -421) and pT_Trig>pT_assoc
          //Excluding self-correlations
          if (particle1.mRowIndex == particle2.mRowIndex) {
            continue;
          }
          entryD0D0barPair(getDeltaPhi(particle2.phi(), particle1.phi()),
                           particle2.eta() - particle1.eta(),
                           particle1.pt(),
                           particle2.pt());
        }
      } // end inner loop (Dbars)
    }   //end outer loop
    registry.fill(HIST("hcountD0D0barPerEvent"), counterD0D0bar);
  }
};

/// c-cbar correlator table builder - for MC gen-level analysis
struct HfCorrelatorCCbarMcGen {

  Produces<aod::DDbarPair> entryccbarPair;

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hPtCandMCGen", "c,cbar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hEtaMCGen", "c,cbar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hYMCGen", "c,cbar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "c,cbar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hcountCCbarPerEvent", "c,cbar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hcountCCbarPerEventPreEtaCut", "c,cbar particles - MC gen;Number per event pre #eta cut;entries", {HistType::kTH1F, {{20, 0., 20.}}}}}};

  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for trigger counters"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hcountCtriggersMCGen", "c trigger particles - MC gen;;N of trigger c quark", {HistType::kTH2F, {{1, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    registry.fill(HIST("hMCEvtCount"), 0);
    int counterccbar = 0, counterccbarPreEtasel = 0;

    //loop over particles at MC gen level
    for (auto& particle1 : particlesMC) {
      if (std::abs(particle1.pdgCode()) != PDG_t::kCharm) { //search c or cbar particles
        continue;
      }
      int partMothPDG = particlesMC.iteratorAt(particle1.mother0()).pdgCode();
      //check whether mothers of quark c/cbar are still '4'/'-4' particles - in that case the c/cbar quark comes from its own fragmentation, skip it
      if (partMothPDG == particle1.pdgCode()) {
        continue;
      }
      counterccbarPreEtasel++; //count c or cbar (before kinematic selection)
      double yC = RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()));
      if (cutYCandMax >= 0. && std::abs(yC) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && particle1.pt() < cutPtCandMin) {
        continue;
      }
      registry.fill(HIST("hPtCandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), yC);
      counterccbar++; //count if c or cbar don't come from themselves during fragmentation (after kinematic selection)

      //c-cbar correlation dedicated section
      //if it's c, search for cbar and evaluate correlations.
      if (particle1.pdgCode() != PDG_t::kCharm) {
        continue;
      }
      registry.fill(HIST("hcountCtriggersMCGen"), 0, particle1.pt()); //to count trigger c quark (for normalisation)

      for (auto& particle2 : particlesMC) {
        if (particle2.pdgCode() != PDG_t::kCharmBar) { //check that inner particle is a cbar
          continue;
        }
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && particle2.pt() < cutPtCandMin) {
          continue;
        }
        //check whether mothers of quark cbar (from associated loop) are still '-4' particles - in that case the cbar quark comes from its own fragmentation, skip it
        if (particlesMC.iteratorAt(particle2.mother0()).pdgCode() == PDG_t::kCharmBar) {
          continue;
        }
        entryccbarPair(getDeltaPhi(particle2.phi(), particle1.phi()),
                       particle2.eta() - particle1.eta(),
                       particle1.pt(),
                       particle2.pt());
      } // end inner loop
    }   //end outer loop
    registry.fill(HIST("hcountCCbarPerEvent"), counterccbar);
    registry.fill(HIST("hcountCCbarPerEventPreEtaCut"), counterccbarPreEtasel);
  }
};

/// c-cbar correlator table builder - for MC gen-level analysis - Like Sign
struct HfCorrelatorCCbarMcGenLs {

  Produces<aod::DDbarPair> entryccbarPair;

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hPtCandMCGen", "c,cbar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hEtaMCGen", "c,cbar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hYMCGen", "c,cbar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "c,cbar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hcountCCbarPerEvent", "c,cbar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hcountCCbarPerEventPreEtaCut", "c,cbar particles - MC gen;Number per event pre #eta cut;entries", {HistType::kTH1F, {{20, 0., 20.}}}}}};

  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for trigger counters"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hcountCtriggersMCGen", "c trigger particles - MC gen;;N of trigger c quark", {HistType::kTH2F, {{1, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    registry.fill(HIST("hMCEvtCount"), 0);
    int counterccbar = 0, counterccbarPreEtasel = 0;

    //loop over particles at MC gen level
    for (auto& particle1 : particlesMC) {
      if (std::abs(particle1.pdgCode()) != PDG_t::kCharm) { //search c or cbar particles
        continue;
      }
      int partMothPDG = particlesMC.iteratorAt(particle1.mother0()).pdgCode();
      //check whether mothers of quark c/cbar are still '4'/'-4' particles - in that case the c/cbar quark comes from its own fragmentation, skip it
      if (partMothPDG == particle1.pdgCode()) {
        continue;
      }
      counterccbarPreEtasel++; //count c or cbar (before kinematic selection)
      double yC = RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()));
      if (cutYCandMax >= 0. && std::abs(yC) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && particle1.pt() < cutPtCandMin) {
        continue;
      }
      registry.fill(HIST("hPtCandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), yC);
      counterccbar++; //count if c or cbar don't come from themselves during fragmentation (after kinematic selection)

      //c-cbar correlation dedicated section
      registry.fill(HIST("hcountCtriggersMCGen"), 0, particle1.pt()); //to count trigger c quark (for normalisation)

      for (auto& particle2 : particlesMC) {
        if (std::abs(particle2.pdgCode()) != PDG_t::kCharm) { //search c or cbar for associated particles
          continue;
        }
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && particle2.pt() < cutPtCandMin) {
          continue;
        }
        if (particle2.pdgCode() == particle1.pdgCode()) {
          //check whether mothers of quark cbar (from associated loop) are still '-4' particles - in that case the cbar quark comes from its own fragmentation, skip it
          if (particlesMC.iteratorAt(particle2.mother0()).pdgCode() == particle2.pdgCode()) {
            continue;
          }
          //Excluding self-correlations
          if (particle1.mRowIndex == particle2.mRowIndex) {
            continue;
          }
          entryccbarPair(getDeltaPhi(particle2.phi(), particle1.phi()),
                         particle2.eta() - particle1.eta(),
                         particle1.pt(),
                         particle2.pt());
        } // end outer if (check PDG associate)
      }   // end inner loop
    }     //end outer loop
    registry.fill(HIST("hcountCCbarPerEvent"), counterccbar);
    registry.fill(HIST("hcountCCbarPerEventPreEtaCut"), counterccbarPreEtasel);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};
  const bool doMCccbar = cfgc.options().get<bool>("doMCccbar");
  const bool doMCGen = cfgc.options().get<bool>("doMCGen");
  const bool doMCRec = cfgc.options().get<bool>("doMCRec");
  const bool doLikeSign = cfgc.options().get<bool>("doLikeSign");
  if (!doLikeSign) { //unlike-sign analyses
    if (doMCGen) {   //MC-Gen analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorD0D0barMcGen>(cfgc));
    } else if (doMCRec) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorD0D0barMcRec>(cfgc));
    } else if (doMCccbar) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorCCbarMcGen>(cfgc));
    } else { //data analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorD0D0bar>(cfgc));
    }
  } else {         //like-sign analyses
    if (doMCGen) { //MC-Gen analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorD0D0barMcGenLs>(cfgc));
    } else if (doMCRec) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorD0D0barMcRecLs>(cfgc));
    } else if (doMCccbar) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorCCbarMcGenLs>(cfgc));
    } else { //data analysis
      workflow.push_back(adaptAnalysisTask<HfCorrelatorD0D0barLs>(cfgc));
    }
  }

  return workflow;
}
