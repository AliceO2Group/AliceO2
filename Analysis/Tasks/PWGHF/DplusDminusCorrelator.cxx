// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DplusDminusCorrelator.cxx
/// \brief Dplus-Dminus correlator task - data-like, MC-reco and MC-kine analyses. For ULS and LS pairs
///
/// \author Fabio Colamaria <fabio.colamaria@ba.infn.it>, INFN Bari

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong3;
using namespace o2::analysis::hf_cuts_dplus_topikpi;
using namespace o2::framework::expressions;
using namespace o2::constants::math;
using namespace o2::aod::hf_ddbar_correlation; 

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
//  ConfigParamSpec optionDoLikeSign{"doLikeSign", VariantType::Bool, false, {"Run Like-Sign analysis."}}; //TOLTO PER ORA
  ConfigParamSpec optionDoMCccbar{"doMCccbar", VariantType::Bool, false, {"Run MC-Gen dedicated tasks."}};
  ConfigParamSpec optionDoMCGen{"doMCGen", VariantType::Bool, false, {"Run MC-Gen dedicated tasks."}};
  ConfigParamSpec optionDoMCRec{"doMCRec", VariantType::Bool, false, {"Run MC-Rec dedicated tasks."}};
//  workflowOptions.push_back(optionDoLikeSign); //TOLTO PER ORA
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

/// definition of variables for DplusDminus pairs vs eta acceptance studies (hDDbarVsEtaCut, in data-like, MC-reco and MC-kine tasks)
const double maxEtaCut = 5.;
const double ptThresholdForMaxEtaCut = 10.;
const double incrementEtaCut = 0.1;
const double incrementPtThreshold = 0.5;
const double epsilon = 1E-5;

/// Dplus-Dminus correlation pair builder - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
struct DplusDminusCorrelator {
  Produces<aod::DDbarPair> entryDplusDminusPair;
  Produces<aod::DDbarRecoInfo> entryDplusDminusRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassDplus for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCand", "Dplus,Dminus candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0", "Dplus,Dminus candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1", "Dplus,Dminus candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatus", "Dplus,Dminus candidates;selection status;entries", {HistType::kTH1F, {{5, -0.5, 4.5}}}},
     {"hEta", "Dplus,Dminus candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhi", "Dplus,Dminus candidates;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hY", "Dplus,Dminus candidates;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDDbarVsEtaCut", "Dplus,Dminus pairs vs #eta cut;#eta_{max};entries", {HistType::kTH2F, {{(int)(maxEtaCut / incrementEtaCut), 0., maxEtaCut}, {(int)(ptThresholdForMaxEtaCut / incrementPtThreshold), 0., ptThresholdForMaxEtaCut}}}}}};

  Configurable<int> dSelectionFlagDplus{"dSelectionFlagDplus", 1, "Selection Flag for Dplus,Dminus"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_dplus_topikpi::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMass", "Dplus,Dminus candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassDplus", "Dplus,Dminus candidates;inv. mass Dplus only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassDminus", "Dplus,Dminus candidates;inv. mass Dminus only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_dplus::isSelDplusToPiKPi >= dSelectionFlagDplus);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelDplusToPiKPiCandidate>> const& candidates, aod::BigTracks const& tracks)
  {
    for (auto& candidate1 : candidates) {
      if (cutYCandMax >= 0. && std::abs(YDPlus(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin) {
        continue;
      }
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << DPlusToPiKPi)) { //probably dummy since already selected? not sure...
        continue;
      }
      int outerParticleSign = 1; //Dplus
      auto outerSecondTrack = candidate1.index1_as<aod::BigTracks>();
      if (outerSecondTrack.sign()==1) {
        outerParticleSign = -1; //Dminus (second daughter track is positive)
      }

      //fill invariant mass plots and generic info from all Dplus/Dminus candidates
      if (outerParticleSign==1) {
        registry.fill(HIST("hMass"), InvMassDPlus(candidate1), candidate1.pt());
        registry.fill(HIST("hMassDplus"), InvMassDPlus(candidate1), candidate1.pt());
      }
      else {
        registry.fill(HIST("hMass"), InvMassDPlus(candidate1), candidate1.pt());
        registry.fill(HIST("hMassDminus"), InvMassDPlus(candidate1), candidate1.pt());
      }
      registry.fill(HIST("hPtCand"), candidate1.pt());
      registry.fill(HIST("hPtProng0"), candidate1.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate1.ptProng1());
      registry.fill(HIST("hEta"), candidate1.eta());
      registry.fill(HIST("hPhi"), candidate1.phi());
      registry.fill(HIST("hY"), YDPlus(candidate1));
      registry.fill(HIST("hSelectionStatus"), candidate1.isSelDplusToPiKPi());

      //D-Dbar correlation dedicated section
      //if the candidate is a Dplus, search for Dminus and evaluate correlations
      if (outerParticleSign==1) {
        for (auto& candidate2 : candidates) {
          //check decay channel flag for candidate2
          if (!(candidate2.hfflag() & 1 << DPlusToPiKPi)) { //probably dummy since already selected? not sure...
            continue;
          }
          int innerParticleSign = 1; //Dplus
          auto innerSecondTrack = candidate2.index1_as<aod::BigTracks>();
          if (innerSecondTrack.sign()==1) {
            innerParticleSign = -1; //Dminus (second daughter track is positive)          
          }
          if (innerParticleSign==-1) {
            if (cutYCandMax >= 0. && std::abs(YDPlus(candidate2)) > cutYCandMax) {
              continue;
            }
            if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin) {
              continue;
            }
            //Excluding trigger self-correlations (possible in case of both mass hypotheses accepted) -> Dummy for D+D-
            if (candidate1.mRowIndex == candidate2.mRowIndex) {
              continue;
            }
            entryDplusDminusPair(getDeltaPhi(candidate2.phi(), candidate1.phi()),
                             candidate2.eta() - candidate1.eta(),
                             candidate1.pt(),
                             candidate2.pt());
            entryDplusDminusRecoInfo(InvMassDPlus(candidate1),
                                 InvMassDPlus(candidate2),
                                 0);
            double etaCut = 0.;
            double ptCut = 0.;
            do { //fill pairs vs etaCut plot
              ptCut = 0.;
              etaCut += incrementEtaCut;
              do { //fill pairs vs etaCut plot
                if (std::abs(candidate1.eta()) < etaCut && std::abs(candidate2.eta()) < etaCut && candidate1.pt() > ptCut && candidate2.pt() > ptCut)
                  registry.fill(HIST("hDDbarVsEtaCut"), etaCut - epsilon, ptCut + epsilon);
                ptCut += incrementPtThreshold;
              } while (ptCut < ptThresholdForMaxEtaCut - epsilon);
            } while (etaCut < maxEtaCut - epsilon);
          }
        } // end inner loop (Dbars)
      }

    } //end outer loop
  }
};

/// Dplus-Dminus correlation pair builder - for MC reco-level analysis (candidates matched to true signal only, but also the various bkg sources are studied)
struct DplusDminusCorrelatorMCRec {
  Produces<aod::DDbarPair> entryDplusDminusPair;
  Produces<aod::DDbarRecoInfo> entryDplusDminusRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassDplus for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCandMCRec", "Dplus,Dminus candidates - MC reco;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0MCRec", "Dplus,Dminus candidates - MC reco;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1MCRec", "Dplus,Dminus candidates - MC reco;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatusMCRec", "Dplus,Dminus candidates - MC reco;selection status;entries", {HistType::kTH1F, {{5, -0.5, 4.5}}}},
     {"hEtaMCRec", "Dplus,Dminus candidates - MC reco;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCRec", "Dplus,Dminus candidates - MC reco;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCRec", "Dplus,Dminus candidates - MC reco;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDDbarVsEtaCut", "Dplus,Dminus pairs vs #eta cut;#eta_{max};entries", {HistType::kTH2F, {{(int)(maxEtaCut / incrementEtaCut), 0., maxEtaCut}, {(int)(ptThresholdForMaxEtaCut / incrementPtThreshold), 0., ptThresholdForMaxEtaCut}}}}}};

  Configurable<int> dSelectionFlagDplus{"dSelectionFlagDplus", 1, "Selection Flag for Dplus,Dminus"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_dplus_topikpi::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMassDplusMCRec", "Dplus,Dminus candidates - MC reco;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassDminusMCRec", "Dplus,Dminus candidates - MC reco;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_dplus::isSelDplusToPiKPi >= dSelectionFlagDplus);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng3, aod::HFSelDplusToPiKPiCandidate, aod::HfCandProng3MCRec>> const& candidates, aod::BigTracks const& tracks)
  {
    //MC reco level
    bool flagDplusSignal = kFALSE;
    bool flagDminusSignal = kFALSE;
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << DPlusToPiKPi)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YDPlus(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin) {
        continue;
      }
      int outerParticleSign = 1; //Dplus
      auto outerSecondTrack = candidate1.index1_as<aod::BigTracks>();
      if (outerSecondTrack.sign()==1) {
        outerParticleSign = -1; //Dminus (second daughter track is positive)
      }
      if (std::abs(candidate1.flagMCMatchRec()) == 1 << DPlusToPiKPi) {
        //fill invariant mass plots and generic info from all Dplus/Dminus candidates
        if (outerParticleSign==1) { //matched as Dplus
          registry.fill(HIST("hMassDplusMCRec"), InvMassDPlus(candidate1), candidate1.pt());
        }
        else { //matched as Dminus
          registry.fill(HIST("hMassDminusMCRec"), InvMassDPlus(candidate1), candidate1.pt());
        }
        registry.fill(HIST("hPtCandMCRec"), candidate1.pt());
        registry.fill(HIST("hPtProng0MCRec"), candidate1.ptProng0());
        registry.fill(HIST("hPtProng1MCRec"), candidate1.ptProng1());
        registry.fill(HIST("hEtaMCRec"), candidate1.eta());
        registry.fill(HIST("hPhiMCRec"), candidate1.phi());
        registry.fill(HIST("hYMCRec"), YDPlus(candidate1));
        registry.fill(HIST("hSelectionStatusMCRec"), candidate1.isSelDplusToPiKPi());
      }

      //D-Dbar correlation dedicated section
      if (outerParticleSign==-1) {
        continue; //reject Dminus in outer loop
      }
      if (std::abs(candidate1.flagMCMatchRec()) == 1 << DPlusToPiKPi) { //candidate matched to Dplus (particle)
        flagDplusSignal = kTRUE;
      } else { //candidate of bkg, wrongly selected as Dplus
        flagDplusSignal = kFALSE;
      }
      for (auto& candidate2 : candidates) {
        if (!(candidate2.hfflag() & 1 << DPlusToPiKPi)) { //check decay channel flag for candidate2
          continue;
        }
        int innerParticleSign = 1; //Dplus
        auto innerSecondTrack = candidate2.index1_as<aod::BigTracks>();
        if (innerSecondTrack.sign()==1) {
          innerParticleSign = -1; //Dminus (second daughter track is positive)          
        }

        if (innerParticleSign==1) {
          continue; //reject Dplus in outer loop
        }        
        if (std::abs(candidate2.flagMCMatchRec()) == 1 << DPlusToPiKPi) { //candidate matched to Dminus (antiparticle)
          flagDminusSignal = kTRUE;
        } else { //candidate of bkg, wrongly selected as Dminus
          flagDminusSignal = kFALSE;
        }
        if (cutYCandMax >= 0. && std::abs(YDPlus(candidate2)) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin) {
          continue;
        }
        //Excluding trigger self-correlations (possible in case of both mass hypotheses accepted) - Seems dummy for D+D-
        if (candidate1.mRowIndex == candidate2.mRowIndex) {
          continue;
        }
        //choice of options (Dplus/Dminus signal/bkg)
        int pairSignalStatus = 0; //0 = bkg/bkg, 1 = bkg/sig, 2 = sig/bkg, 3 = sig/sig
        if (flagDplusSignal) {
          pairSignalStatus += 2;
        }
        if (flagDminusSignal) {
          pairSignalStatus += 1;
        }
        entryDplusDminusPair(getDeltaPhi(candidate2.phi(), candidate1.phi()),
                         candidate2.eta() - candidate1.eta(),
                         candidate1.pt(),
                         candidate2.pt());
        entryDplusDminusRecoInfo(InvMassDPlus(candidate1),
                             InvMassDPlus(candidate2),
                             pairSignalStatus);
        double etaCut = 0.;
        double ptCut = 0.;
        do { //fill pairs vs etaCut plot
          ptCut = 0.;
          etaCut += incrementEtaCut;
          do { //fill pairs vs etaCut plot
            if (std::abs(candidate1.eta()) < etaCut && std::abs(candidate2.eta()) < etaCut && candidate1.pt() > ptCut && candidate2.pt() > ptCut)
              registry.fill(HIST("hDDbarVsEtaCut"), etaCut - epsilon, ptCut + epsilon);
            ptCut += incrementPtThreshold;
          } while (ptCut < ptThresholdForMaxEtaCut - epsilon);
        } while (etaCut < maxEtaCut - epsilon);
      } // end inner loop (Dbars)

    } //end outer loop
  }
};

/// Dplus-Dminus correlation pair builder - for MC gen-level analysis (no filter/selection, only true signal)
struct DplusDminusCorrelatorMCGen {

  Produces<aod::DDbarPair> entryDplusDminusPair;

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hPtCandMCGen", "Dplus,Dminus particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hEtaMCGen", "Dplus,Dminus particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "Dplus,Dminus particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCGen", "Dplus,Dminus candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hcountDplusDminusPerEvent", "Dplus,Dminus particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hDDbarVsEtaCut", "Dplus,Dminus pairs vs #eta cut;#eta_{max};entries", {HistType::kTH2F, {{(int)(maxEtaCut / incrementEtaCut), 0., maxEtaCut}, {(int)(ptThresholdForMaxEtaCut / incrementPtThreshold), 0., ptThresholdForMaxEtaCut}}}}}};

  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_dplus_topikpi::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hcountDplustriggersMCGen", "Dplus trigger particles - MC gen;;N of trigger D0", {HistType::kTH2F, {{1, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng3MCGen> const& particlesMC)
  {
    int counterDplusDminus = 0;
    registry.fill(HIST("hMCEvtCount"), 0);
    //MC gen level
    for (auto& particle1 : particlesMC) {
      //check if the particle is Dplus or Dminus (for general plot filling and selection, so both cases are fine) - NOTE: decay channel is not probed!
      if (std::abs(particle1.pdgCode()) != 411) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()))) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(particle1.pt()) < cutPtCandMin) {
        continue;
      }
      registry.fill(HIST("hPtCandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode())));
      counterDplusDminus++;

      //D-Dbar correlation dedicated section
      //if it's a Dplus particle, search for Dminus and evaluate correlations
      if (particle1.pdgCode() == 411) {                                  //just checking the particle PDG, not the decay channel (differently from Reco: you have a BR factor btw such levels!)
        registry.fill(HIST("hcountDplustriggersMCGen"), 0, particle1.pt()); //to count trigger Dplus (for normalisation)
        for (auto& particle2 : particlesMC) {
          if (particle2.pdgCode() == -411) {
            if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
              continue;
            }
            if (cutPtCandMin >= 0. && std::abs(particle2.pt()) < cutPtCandMin) {
              continue;
            }
            entryDplusDminusPair(getDeltaPhi(particle2.phi(), particle1.phi()),
                             particle2.eta() - particle1.eta(),
                             particle1.pt(),
                             particle2.pt());
            double etaCut = 0.;
            double ptCut = 0.;
            do { //fill pairs vs etaCut plot
              ptCut = 0.;
              etaCut += incrementEtaCut;
              do { //fill pairs vs etaCut plot
                if (std::abs(particle1.eta()) < etaCut && std::abs(particle2.eta()) < etaCut && particle1.pt() > ptCut && particle2.pt() > ptCut)
                  registry.fill(HIST("hDDbarVsEtaCut"), etaCut - epsilon, ptCut + epsilon);
                ptCut += incrementPtThreshold;
              } while (ptCut < ptThresholdForMaxEtaCut - epsilon);
            } while (etaCut < maxEtaCut - epsilon);
          } // end Dminus check
        }   //end inner loop
      }     //end Dplus check

    } //end outer loop
    registry.fill(HIST("hcountDplusDminusPerEvent"), counterDplusDminus);
  }
};
/*
/// D0-D0bar correlation pair builder - LIKE SIGN - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
struct DDbarCorrelatorLS {

  Produces<aod::DDbarPair> entryD0D0barPair;
  Produces<aod::DDbarRecoInfo> entryD0D0barRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCand", "D0,D0bar candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0", "D0,D0bar candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1", "D0,D0bar candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatus", "D0,D0bar candidates;selection status;entries", {HistType::kTH1F, {{5, -0.5, 4.5}}}},
     {"hEta", "D0,D0bar candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhi", "D0,D0bar candidates;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hY", "D0,D0bar candidates;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}}}};

  Configurable<int> dSelectionFlagD0{"dSelectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> dSelectionFlagD0bar{"dSelectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMass", "D0,D0bar candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0", "D0,D0bar candidates;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0bar", "D0,D0bar candidates;inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= dSelectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= dSelectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate>> const& candidates)
  {
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin) {
        continue;
      }
      //fill invariant mass plots and generic info from all D0/D0bar candidates
      if (candidate1.isSelD0() >= dSelectionFlagD0) {
        registry.fill(HIST("hMass"), InvMassD0(candidate1), candidate1.pt());
        registry.fill(HIST("hMassD0"), InvMassD0(candidate1), candidate1.pt());
      }
      if (candidate1.isSelD0bar() >= dSelectionFlagD0bar) {
        registry.fill(HIST("hMass"), InvMassD0bar(candidate1), candidate1.pt());
        registry.fill(HIST("hMassD0bar"), InvMassD0bar(candidate1), candidate1.pt());
      }
      registry.fill(HIST("hPtCand"), candidate1.pt());
      registry.fill(HIST("hPtProng0"), candidate1.ptProng0());
      registry.fill(HIST("hPtProng1"), candidate1.ptProng1());
      registry.fill(HIST("hEta"), candidate1.eta());
      registry.fill(HIST("hPhi"), candidate1.phi());
      registry.fill(HIST("hY"), YD0(candidate1));
      registry.fill(HIST("hSelectionStatus"), candidate1.isSelD0() + (candidate1.isSelD0bar() * 2));

      double ptParticle1 = candidate1.pt(); //trigger particle is the largest-pT one

      //D-Dbar correlation dedicated section
      //For like-sign, first loop on both D0 and D0bars. First candidate is for sure a D0 and D0bars (checked before, so don't re-check anything on it)
      for (auto& candidate2 : candidates) {
        //check decay channel flag for candidate2
        if (!(candidate2.hfflag() & 1 << D0ToPiK)) {
          continue;
        }
        //for the associated, has to have smaller pT, and pass D0sel if trigger passes D0sel, or D0barsel if trigger passes D0barsel
        if (candidate2.pt() < ptParticle1 && ((candidate1.isSelD0() >= dSelectionFlagD0 && candidate2.isSelD0() >= dSelectionFlagD0) || (candidate1.isSelD0bar() >= dSelectionFlagD0bar && candidate2.isSelD0bar() >= dSelectionFlagD0bar))) {
          if (cutYCandMax >= 0. && std::abs(YD0(candidate2)) > cutYCandMax) {
            continue;
          }
          if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin) {
            continue;
          }
          //Excluding self-correlations (in principle not possible due to the '<' condition, but could rounding break it?)
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

    } //end outer loop
  }
};

/// D0-D0bar correlation pair builder - LIKE SIGN - for MC reco analysis (data-like but matching to true DO and D0bar)
struct D0D0barCorrelatorMCRecLS {

  Produces<aod::DDbarPair> entryD0D0barPair;
  Produces<aod::DDbarRecoInfo> entryD0D0barRecoInfo;

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 for trigger normalisation (S*0.955), and hMass2DCorrelationPairs (in final task) for 2D-sideband-subtraction purposes
    {{"hPtCandMCRec", "D0,D0bar candidates - MC reco;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng0MCRec", "D0,D0bar candidates - MC reco;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hPtProng1MCRec", "D0,D0bar candidates - MC reco;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hSelectionStatusMCRec", "D0,D0bar candidates - MC reco;selection status;entries", {HistType::kTH1F, {{5, -0.5, 4.5}}}},
     {"hEtaMCRec", "D0,D0bar candidates - MC reco;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCRec", "D0,D0bar candidates - MC reco;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCRec", "D0,D0bar candidates - MC reco;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}}}};

  Configurable<int> dSelectionFlagD0{"dSelectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> dSelectionFlagD0bar{"dSelectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for candidate mass plots"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hMassD0MCRec", "D0,D0bar candidates - MC reco;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
    registry.add("hMassD0barMCRec", "D0,D0bar candidates - MC reco;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{120, 1.5848, 2.1848}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= dSelectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= dSelectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HfCandProng2MCRec>> const& candidates)
  {
    //MC reco level
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin) {
        continue;
      }
      if (std::abs(candidate1.flagMCMatchRec()) == 1 << D0ToPiK) {
        //fill invariant mass plots and generic info from all D0/D0bar candidates
        if (candidate1.isSelD0() >= dSelectionFlagD0 && candidate1.flagMCMatchRec() == D0ToPiK) { //only reco and matched as D0
          registry.fill(HIST("hMassD0MCRec"), InvMassD0(candidate1));
        }
        if (candidate1.isSelD0bar() >= dSelectionFlagD0bar && candidate1.flagMCMatchRec() == D0ToPiK) { //only reco and matched as D0bar
          registry.fill(HIST("hMassD0barMCRec"), InvMassD0bar(candidate1));
        }
        registry.fill(HIST("hPtCandMCRec"), candidate1.pt());
        registry.fill(HIST("hPtprong0MCRec"), candidate1.ptProng0());
        registry.fill(HIST("hPtprong1MCRec"), candidate1.ptProng1());
        registry.fill(HIST("hEtaMCRec"), candidate1.eta());
        registry.fill(HIST("hPhiMCRec"), candidate1.phi());
        registry.fill(HIST("hYMCRec"), YD0(candidate1));

        double ptParticle1 = candidate1.pt(); //trigger particle is the largest pT one

        //D-Dbar correlation dedicated section
        //For like-sign, first loop on both D0 and D0bars. First candidate is for sure a D0 and D0bars (looping on filtered) and was already matched, so don't re-check anything on it)
        for (auto& candidate2 : candidates) {
          //check decay channel flag for candidate2
          if (!(candidate2.hfflag() & 1 << D0ToPiK)) {
            continue;
          }
          bool conditionLSForD0 = (candidate1.isSelD0() >= dSelectionFlagD0bar && candidate1.flagMCMatchRec() == 1 << D0ToPiK) && (candidate2.isSelD0() >= dSelectionFlagD0bar && candidate2.flagMCMatchRec() == 1 << D0ToPiK);
          bool conditionLSForD0bar = (candidate1.isSelD0bar() >= dSelectionFlagD0bar && candidate1.flagMCMatchRec() == -1 << D0ToPiK) && (candidate2.isSelD0bar() >= dSelectionFlagD0bar && candidate2.flagMCMatchRec() == -1 << D0ToPiK);
          if (candidate2.pt() < ptParticle1 && (conditionLSForD0 || conditionLSForD0bar)) { //LS pair (of D0 or of D0bar) + pt2<pt1
            if (cutYCandMax >= 0. && std::abs(YD0(candidate2)) > cutYCandMax) {
              continue;
            }
            if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin) {
              continue;
            }
            //Excluding self-correlations (in principle not possible due to the '<' condition, but could rounding break it?)
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

        } // end inner loop (Dbars)
      }   //end outer if (MC match)
    }     //end outer loop
  }
};

/// D0-D0bar correlation pair builder - for MC gen-level analysis, like sign particles
struct D0D0barCorrelatorMCGenLS {

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
      if (std::abs(particle1.pdgCode()) != 421) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()))) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(particle1.pt()) < cutPtCandMin) {
        continue;
      }

      double ptParticle1 = particle1.pt(); //trigger particle is the largest pT one

      registry.fill(HIST("hPtCandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode())));
      counterD0D0bar++;
      //D-Dbar correlation dedicated section
      //if it's D0, search for D0bar and evaluate correlations.
      registry.fill(HIST("hcountD0triggersMCGen"), 0, particle1.pt()); //to count trigger D0 (normalisation)
      for (auto& particle2 : particlesMC) {
        if (std::abs(particle2.pdgCode()) != 421) { //check that associated is a D0/D0bar (both are fine)
          continue;
        }
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && std::abs(particle2.pt()) < cutPtCandMin) {
          continue;
        }
        if (particle2.pt() < ptParticle1 && particle2.pdgCode() == particle1.pdgCode()) { //like-sign condition (both 421 or both -421) and pT_Trig>pT_assoc
          //Excluding self-correlations (in principle not possible due to the '<' condition, but could rounding break it?)
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
*/
/// c-cbar correlator table builder - for MC gen-level analysis
struct CCbarCorrelatorMCGen {

  Produces<aod::DDbarPair> entryD0D0barPair;

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hPtCandMCGen", "c,cbar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hEtaMCGen", "c,cbar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hYMCGen", "c,cbar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "c,cbar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hcountD0D0barPerEvent", "D0,cbar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}}}};

  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for trigger counters"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hcountD0triggersMCGen", "c trigger particles - MC gen;;N of trigger D0", {HistType::kTH2F, {{1, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    registry.fill(HIST("hMCEvtCount"), 0);
    int counterccbar = 0, counterccbarPreEtasel = 0;

    //loop over particles at MC gen level
    for (auto& particle1 : particlesMC) {
      if (std::abs(particle1.pdgCode()) != 4) { //search c or cbar particles
        continue;
      }
      int partMothPDG = particlesMC.iteratorAt(particle1.mother0()).pdgCode();
      //check whether mothers of quark c/cbar are still '4'/'-4' particles - in that case the c/cbar quark comes from its own fragmentation, skip it
      if (partMothPDG == particle1.pdgCode()) {
        continue;
      }
      counterccbarPreEtasel++; //count c or cbar (before kinematic selection)
      if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()))) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(particle1.pt()) < cutPtCandMin) {
        continue;
      }
      registry.fill(HIST("hPtcandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode())));
      counterccbar++; //count if c or cbar don't come from themselves during fragmentation (after kinematic selection)

      //c-cbar correlation dedicated section
      //if it's c, search for cbar and evaluate correlations.
      if (particle1.pdgCode() == 4) {

        registry.fill(HIST("hcountD0triggersMCGen"), 0, particle1.pt()); //to count trigger c quark (for normalisation)

        for (auto& particle2 : particlesMC) {
          if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
            continue;
          }
          if (cutPtCandMin >= 0. && std::abs(particle2.pt()) < cutPtCandMin) {
            continue;
          }
          if (particle2.pdgCode() == -4) {
            //check whether mothers of quark cbar (from associated loop) are still '-4' particles - in that case the cbar quark comes from its own fragmentation, skip it
            if (particlesMC.iteratorAt(particle2.mother0()).pdgCode() == -4) {
              continue;
            }
            entryD0D0barPair(getDeltaPhi(particle2.phi(), particle1.phi()),
                             particle2.eta() - particle1.eta(),
                             particle1.pt(),
                             particle2.pt());
          } // end outer if (check cbar)
        }   // end inner loop
      }     //end outer if (check c)
    }       //end outer loop
    registry.fill(HIST("hcountCCbarPerEvent"), counterccbar);
    registry.fill(HIST("hcountCCbarPerEventPreEtaCut"), counterccbarPreEtasel);
  }
};
/*
/// c-cbar correlator table builder - for MC gen-level analysis - Like Sign
struct CCbarCorrelatorMCGenLS {

  Produces<aod::DDbarPair> entryD0D0barPair;

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hPtCandMCGen", "c,cbar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{180, 0., 36.}}}},
     {"hEtaMCGen", "c,cbar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hYMCGen", "c,cbar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "c,cbar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hcountD0D0barPerEvent", "D0,cbar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}}}};

  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};
  Configurable<std::vector<double>> bins{"ptBinsForMass", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for trigger counters"};

  void init(o2::framework::InitContext&)
  {
    registry.add("hcountD0triggersMCGen", "c trigger particles - MC gen;;N of trigger D0", {HistType::kTH2F, {{1, -0.5, 0.5}, {(std::vector<double>)bins, "#it{p}_{T} (GeV/#it{c})"}}});
  }

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    registry.fill(HIST("hMCEvtCount"), 0);
    int counterccbar = 0, counterccbarPreEtasel = 0;

    //loop over particles at MC gen level
    for (auto& particle1 : particlesMC) {
      if (std::abs(particle1.pdgCode()) != 4) { //search c or cbar particles
        continue;
      }
      int partMothPDG = particlesMC.iteratorAt(particle1.mother0()).pdgCode();
      //check whether mothers of quark c/cbar are still '4'/'-4' particles - in that case the c/cbar quark comes from its own fragmentation, skip it
      if (partMothPDG == particle1.pdgCode()) {
        continue;
      }
      counterccbarPreEtasel++; //count c or cbar (before kinematic selection)
      if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode()))) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && std::abs(particle1.pt()) < cutPtCandMin) {
        continue;
      }
      registry.fill(HIST("hPtcandMCGen"), particle1.pt());
      registry.fill(HIST("hEtaMCGen"), particle1.eta());
      registry.fill(HIST("hPhiMCGen"), particle1.phi());
      registry.fill(HIST("hYMCGen"), RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode())));
      counterccbar++; //count if c or cbar don't come from themselves during fragmentation (after kinematic selection)

      //c-cbar correlation dedicated section
      double ptParticle1 = particle1.pt();                             //trigger particle is the largest pT one
      registry.fill(HIST("hcountD0triggersMCGen"), 0, particle1.pt()); //to count trigger c quark (for normalisation)

      for (auto& particle2 : particlesMC) {
        if (std::abs(particle2.pdgCode()) != 4) { //search c or cbar for associated particles
          continue;
        }
        if (cutYCandMax >= 0. && std::abs(RecoDecay::Y(array{particle2.px(), particle2.py(), particle2.pz()}, RecoDecay::getMassPDG(particle2.pdgCode()))) > cutYCandMax) {
          continue;
        }
        if (cutPtCandMin >= 0. && std::abs(particle2.pt()) < cutPtCandMin) {
          continue;
        }
        if (particle2.pt() < ptParticle1 && particle2.pdgCode() == particle1.pdgCode()) {
          //check whether mothers of quark cbar (from associated loop) are still '-4' particles - in that case the cbar quark comes from its own fragmentation, skip it
          if (particlesMC.iteratorAt(particle2.mother0()).pdgCode() == particle2.pdgCode()) {
            continue;
          }
          entryD0D0barPair(getDeltaPhi(particle2.phi(), particle1.phi()),
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
*/
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};
  const bool doMCccbar = cfgc.options().get<bool>("doMCccbar");
  const bool doMCGen = cfgc.options().get<bool>("doMCGen");
  const bool doMCRec = cfgc.options().get<bool>("doMCRec");
//  const bool doLikeSign = cfgc.options().get<bool>("doLikeSign");
//  if (!doLikeSign) { //unlike-sign analyses
    if (doMCGen) {   //MC-Gen analysis
      workflow.push_back(adaptAnalysisTask<DplusDminusCorrelatorMCGen>(cfgc, TaskName{"dplusdminus-correlator-mc-gen"}));
    } else if (doMCRec) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<DplusDminusCorrelatorMCRec>(cfgc, TaskName{"dplusdminus-correlator-mc-rec"}));
    } else if (doMCccbar) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<CCbarCorrelatorMCGen>(cfgc, TaskName{"ccbar-correlator-mc-gen"}));
    } else { //data analysis
      workflow.push_back(adaptAnalysisTask<DplusDminusCorrelator>(cfgc, TaskName{"dplusdminus-correlator"}));
    }
//}
   /* else {         //like-sign analyses //FABIO -> QUESTE NON MODIFICATE!
    if (doMCGen) { //MC-Gen analysis
      workflow.push_back(adaptAnalysisTask<D0D0barCorrelatorMCGenLS>(cfgc, TaskName{"d0d0bar-correlator-mc-gen-ls"}));
    } else if (doMCRec) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<D0D0barCorrelatorMCRecLS>(cfgc, TaskName{"d0d0bar-correlator-mc-rec-ls"}));
    } else if (doMCccbar) { //MC-Reco analysis
      workflow.push_back(adaptAnalysisTask<CCbarCorrelatorMCGenLS>(cfgc, TaskName{"ccbar-correlator-mc-gen-ls"}));
    } else { //data analysis
      workflow.push_back(adaptAnalysisTask<D0D0barCorrelatorLS>(cfgc, TaskName{"d0d0bar-correlator-ls"}));
    }
  }
*/
  return workflow;
}
