// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskD0D0barCorrelation.cxx
/// \brief D0-D0bar analysis task - data-like, MC-reco and MC-kine analyses. For ULS and LS pairs
///
/// \author Fabio Colamaria <fabio.colamaria@ba.infn.it>, INFN Bari

#include <cmath>

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::framework::expressions;
using namespace o2::constants::math;

// the following is needed to extend the standard candidate table, and allow grouping candidate table by collisions
namespace o2::aod
{
namespace hf_2prong_correlation
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
} // namespace hf_2prong_correlation
DECLARE_SOA_TABLE(HF2ProngCollis, "AOD", "COLLID_2PR", aod::hf_2prong_correlation::CollisionId);

using Big2Prong = soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HF2ProngCollis>;
using Big2ProngMC = soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HfCandProng2MCRec, aod::HF2ProngCollis>;
} // namespace o2::aod

// preliminary task to fill the column index to the extended candidate table
struct CreateBig2Prong {

  Produces<aod::HF2ProngCollis> create2ProngIndexCollColumn;
  void process(aod::HfCandProng2 const& candidates, aod::Tracks const& tracks)
  {
    for (auto& candidate : candidates) {
      int indexColl = candidate.index0_as<aod::Tracks>().collisionId(); //takes index of collision from first D daughter
      create2ProngIndexCollColumn(indexColl);
    }
  }
};

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Run MC-dedicated tasks."}};
  workflowOptions.push_back(optionDoMC);
}

#include "Framework/runDataProcessing.h"

double getDeltaPhi(double phiD, double phiDbar)
{
  double dPhi = phiDbar - phiD;

  if (dPhi < -o2::constants::math::PI / 2.)
    dPhi = dPhi + 2. * o2::constants::math::PI;
  if (dPhi > 3. * o2::constants::math::PI / 2.)
    dPhi = dPhi - 2. * o2::constants::math::PI;

  return dPhi;
}

double evaluatePhiByVertex(double xVertex1, double xVertex2, double yVertex1, double yVertex2)
{
  double phi = std::atan2((yVertex2 - yVertex1), (xVertex2 - xVertex1));

  if (phi < 0.)
    phi = phi + 2. * o2::constants::math::PI;
  if (phi > 2. * o2::constants::math::PI)
    phi = phi - 2. * o2::constants::math::PI;

  return phi;
}

/// D0 analysis task - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
struct TaskD0D0barCorrelation {

  double maxEtaCut = 5., PtThresholdForMaxEtaCut = 10.; //for hDDbarVsEtaCut, gives eta increment of 0.1 and pt thr increments of 0.5

  HistogramRegistry registry{
    "registry",
    //NOTE: use hmassD0 for normalisation, and hmass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hmass", "D0,D0bar candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{200, 1., 3.}}}},
     {"hmassD0", "D0,D0bar candidates;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{200, 1., 3.}}}},
     {"hmassD0bar", "D0,D0bar candidates;inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{200, 1., 3.}}}},
     {"hmass2DCorrelationPairs", "D0,D0bar candidates 2D;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{200, 1., 3.}, {200, 1., 3.}}}},
     {"hptcand", "D0,D0bar candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0", "D0,D0bar candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1", "D0,D0bar candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hselectionstatus", "D0,D0bar candidates;selection status;entries", {HistType::kTH1F, {{5, -0.5, 4.5}}}},
     {"hEta", "D0,D0bar candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhi", "D0,D0bar candidates;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hY", "D0,D0bar candidates;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDeltaEtaPtInt", "D0,D0bar candidates;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtInt", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtInt", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaEtaVsPt", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {200, -10., 10.}}}},
     {"hDeltaPhiVsPt", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPtDDbar", "D0,D0bar candidates;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMin", "D0,D0bar candidates;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hDDbarVsEtaCut", "D0,D0bar pairs vs #eta cut;#eta_{max};entries", {HistType::kTH2F, {{(int)(10 * maxEtaCut), 0., maxEtaCut}, {(int)(2 * PtThresholdForMaxEtaCut), 0., PtThresholdForMaxEtaCut}}}}}}; //don't modify the binning from here: act on maxEtaCut and PtThresholdForMaxEtaCut instead

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<aod::Big2Prong> const& candidates, aod::BigTracks const& tracks)
  {
    for (auto& candidate1 : candidates) {
      if (cutEtaCandMax >= 0. && std::abs(candidate1.eta()) > cutEtaCandMax)
        continue;
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin)
        continue;
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      //fill invariant mass plots and generic info from all D0/D0bar candidates
      if (candidate1.isSelD0() >= d_selectionFlagD0) {
        registry.fill(HIST("hmass"), InvMassD0(candidate1));
        registry.fill(HIST("hmassD0"), InvMassD0(candidate1));
      }
      if (candidate1.isSelD0bar() >= d_selectionFlagD0bar) {
        registry.fill(HIST("hmass"), InvMassD0bar(candidate1));
        registry.fill(HIST("hmassD0bar"), InvMassD0bar(candidate1));
      }
      registry.fill(HIST("hptcand"), candidate1.pt());
      registry.fill(HIST("hptprong0"), candidate1.ptProng0());
      registry.fill(HIST("hptprong1"), candidate1.ptProng1());
      registry.fill(HIST("hEta"), candidate1.eta());
      registry.fill(HIST("hPhi"), candidate1.phi());
      registry.fill(HIST("hY"), YD0(candidate1));
      registry.fill(HIST("hselectionstatus"), candidate1.isSelD0() + (candidate1.isSelD0bar() * 2));

      //D-Dbar correlation dedicated section
      //if the candidate is a D0, search for D0bar and evaluate correlations
      if (candidate1.isSelD0() >= d_selectionFlagD0) {
        for (auto& candidate2 : candidates) {
          //check decay channel flag for candidate2
          if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
            continue;
          }
          if (candidate2.isSelD0bar() >= d_selectionFlagD0bar) {
            if (cutEtaCandMax >= 0. && std::abs(candidate2.eta()) > cutEtaCandMax)
              continue;
            if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin)
              continue;
            //Excluding trigger self-correlations (possible in case of both mass hypotheses accepted)
            if (candidate1.mRowIndex == candidate2.mRowIndex)
              continue;
            double eta1 = candidate1.eta(), eta2 = candidate2.eta(), pt1 = candidate1.pt(), pt2 = candidate2.pt(), phi1 = candidate1.phi(), phi2 = candidate2.phi();
            registry.fill(HIST("hmass2DCorrelationPairs"), InvMassD0(candidate1), InvMassD0bar(candidate2));
            registry.fill(HIST("hDeltaEtaPtInt"), eta2 - eta1);
            registry.fill(HIST("hDeltaPhiPtInt"), getDeltaPhi(phi2, phi1));
            registry.fill(HIST("hCorrel2DPtInt"), getDeltaPhi(phi2, phi1), eta2 - eta1);
            registry.fill(HIST("hDeltaEtaVsPt"), pt1, pt2, eta2 - eta1);
            registry.fill(HIST("hDeltaPhiVsPt"), pt1, pt2, getDeltaPhi(phi2, phi1));
            registry.fill(HIST("hDeltaPtDDbar"), pt2 - pt1);
            registry.fill(HIST("hDeltaPtMaxMin"), std::max(pt2, pt1) - std::min(pt2, pt1));
            double etaCut = 0.;
            double ptCut = 0.;
            do { //fill pairs vs etaCut plot
              ptCut = 0.;
              etaCut += 0.1;
              do { //fill pairs vs etaCut plot
                if (std::abs(candidate1.eta()) < etaCut && std::abs(candidate2.eta()) < etaCut && candidate1.pt() > ptCut && candidate2.pt() > ptCut)
                  registry.fill(HIST("hDDbarVsEtaCut"), etaCut - 0.01, ptCut + 0.01);
                ptCut += 0.5;
              } while (ptCut < PtThresholdForMaxEtaCut - 0.05);
            } while (etaCut < maxEtaCut - 0.05);
          }
          //note: candidates selected as both D0 and D0bar are used, and considered in both situation (but not auto-correlated): reflections could play a relevant role.
          //another, more restrictive, option, could be to consider only candidates selected with a single option (D0 xor D0bar)

        } // end inner loop (Dbars)
      }

    } //end outer loop
  }
};

/// D0 analysis task - for MC reco-level analysis (candidates matched to true signal only)
struct TaskD0D0barCorrelationMCRec {

  double maxEtaCut = 5., PtThresholdForMaxEtaCut = 10.; //for hDDbarVsEtaCut, gives eta increment of 0.1 and pt thr increments of 0.5

  HistogramRegistry registry{
    "registry",
    //NOTE: use hmassD0MCRec for normalisation, and hmass2DCorrelationPairsMCRec for 2D-sideband-subtraction purposes
    {{"hmassD0MCRec", "D0,D0bar candidates - MC reco;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmassD0barMCRec", "D0,D0bar candidates - MC reco;inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmass2DCorrelationPairsMCRec", "D0,D0bar candidates 2D;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{200, 1., 3.}, {200, 1., 3.}}}},
     {"hptcandMCRec", "D0,D0bar candidates - MC reco;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0MCRec", "D0,D0bar candidates - MC reco;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1MCRec", "D0,D0bar candidates - MC reco;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hEtaMCRec", "D0,D0bar candidates - MC reco;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCRec", "D0,D0bar candidates - MC reco;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCRec", "D0,D0bar candidates - MC reco;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDeltaEtaPtIntMCRec", "D0,D0bar candidates - MC reco;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCRec", "D0,D0bar candidates - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCRec", "D0,D0bar candidates - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaEtaVsPtMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {200, -10., 10.}}}},
     {"hDeltaPhiVsPtMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPtDDbarMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hDDbarVsEtaCutMCRec", "D0,D0bar pairs vs #eta cut - MC reco;#eta_{max};entries", {HistType::kTH2F, {{(int)(10 * maxEtaCut), 0., maxEtaCut}, {(int)(2 * PtThresholdForMaxEtaCut), 0., PtThresholdForMaxEtaCut}}}}}}; //don't modify the binning from here: act on maxEtaCut and PtThresholdForMaxEtaCut instead

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<aod::Big2ProngMC> const& candidates, aod::BigTracks const& tracks)
  {
    //MC reco level
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutEtaCandMax >= 0. && std::abs(candidate1.eta()) > cutEtaCandMax)
        continue;
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin)
        continue;
      if (std::abs(candidate1.flagMCMatchRec()) == 1 << D0ToPiK) {
        //fill invariant mass plots and generic info from all D0/D0bar candidates
        if (candidate1.isSelD0() >= d_selectionFlagD0 && candidate1.flagMCMatchRec() == 1 << D0ToPiK) { //only reco and matched as D0
          registry.fill(HIST("hmassD0MCRec"), InvMassD0(candidate1));
        }
        if (candidate1.isSelD0bar() >= d_selectionFlagD0bar && candidate1.flagMCMatchRec() == -1 << D0ToPiK) { //only reco and matched as D0bar
          registry.fill(HIST("hmassD0barMCRec"), InvMassD0bar(candidate1));
        }
        registry.fill(HIST("hptcandMCRec"), candidate1.pt());
        registry.fill(HIST("hptprong0MCRec"), candidate1.ptProng0());
        registry.fill(HIST("hptprong1MCRec"), candidate1.ptProng1());
        registry.fill(HIST("hEtaMCRec"), candidate1.eta());
        registry.fill(HIST("hPhiMCRec"), candidate1.phi());
        registry.fill(HIST("hYMCRec"), YD0(candidate1));

        //D-Dbar correlation dedicated section
        //if the candidate is a D0, search for D0bar and evaluate correlations
        if (candidate1.isSelD0() >= d_selectionFlagD0 && candidate1.flagMCMatchRec() == 1 << D0ToPiK) { //selected as D0 (particle) && matched to D0 (particle)
          for (auto& candidate2 : candidates) {
            //check decay channel flag for candidate2
            if (!(candidate2.hfflag() & 1 << D0ToPiK)) {
              continue;
            }
            if (candidate2.isSelD0bar() >= d_selectionFlagD0bar && candidate2.flagMCMatchRec() == -1 << D0ToPiK) { //selected as D0bar (antiparticle) && matched to D0bar (antiparticle)
              if (cutEtaCandMax >= 0. && std::abs(candidate2.eta()) > cutEtaCandMax)
                continue;
              if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin)
                continue;
              double eta1 = candidate1.eta(), eta2 = candidate2.eta(), pt1 = candidate1.pt(), pt2 = candidate2.pt(), phi1 = candidate1.phi(), phi2 = candidate2.phi();
              registry.fill(HIST("hmass2DCorrelationPairsMCRec"), InvMassD0(candidate1), InvMassD0bar(candidate2));
              registry.fill(HIST("hDeltaEtaPtIntMCRec"), eta2 - eta1);
              registry.fill(HIST("hDeltaPhiPtIntMCRec"), getDeltaPhi(phi2, phi1));
              registry.fill(HIST("hCorrel2DPtIntMCRec"), getDeltaPhi(phi2, phi1), eta2 - eta1);
              registry.fill(HIST("hDeltaEtaVsPtMCRec"), pt1, pt2, eta2 - eta1);
              registry.fill(HIST("hDeltaPhiVsPtMCRec"), pt1, pt2, getDeltaPhi(phi2, phi1));
              registry.fill(HIST("hDeltaPtDDbarMCRec"), pt2 - pt1);
              registry.fill(HIST("hDeltaPtMaxMinMCRec"), std::max(pt2, pt1) - std::min(pt2, pt1));
              double etaCut = 0.;
              double ptCut = 0.;
              do { //fill pairs vs etaCut plot
                ptCut = 0.;
                etaCut += 0.1;
                do { //fill pairs vs etaCut plot
                  if (std::abs(candidate1.eta()) < etaCut && std::abs(candidate2.eta()) < etaCut && candidate1.pt() > ptCut && candidate2.pt() > ptCut)
                    registry.fill(HIST("hDDbarVsEtaCutMCRec"), etaCut - 0.01, ptCut + 0.01);
                  ptCut += 0.5;
                } while (ptCut < PtThresholdForMaxEtaCut - 0.05);
              } while (etaCut < maxEtaCut - 0.05);
            } //end inner if (MC match)

          } // end inner loop (Dbars)
        }
      } //end outer if (MC match)
    }   //end outer loop
  }
};

/// D0 analysis task - for MC gen-level analysis (no filter/selection, only true signal)
struct TaskD0D0barCorrelationMCGen {

  double maxEtaCut = 5., PtThresholdForMaxEtaCut = 10.; //for hDDbarVsEtaCut, gives eta increment of 0.1 and pt thr increments of 0.5

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hptcandMCGen", "D0,D0bar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hcountD0triggersMCGen", "D0 trigger particles - MC gen;;N of trigger D0", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hEtaMCGen", "D0,D0bar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "D0,D0bar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCGen", "D0,D0bar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDeltaEtaPtIntMCGen", "D0,D0bar particles - MC gen;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCGen", "D0,D0bar particles - MC gen;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCGen", "D0,D0bar particles - MC gen;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaEtaVsPtMCGen", "D0,D0bar candidates - MC gen;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {200, -10., 10.}}}},
     {"hDeltaPhiVsPtMCGen", "D0,D0bar candidates - MC gen;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPtDDbarMCGen", "D0,D0bar particles - MC gen;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCGen", "D0,D0bar particles - MC gen;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hDDbarVsEtaCutMCGen", "D0,D0bar pairs vs #eta cut - MC gen;#eta_{max};entries", {HistType::kTH2F, {{(int)(10 * maxEtaCut), 0., maxEtaCut}, {(int)(2 * PtThresholdForMaxEtaCut), 0., PtThresholdForMaxEtaCut}}}}, //don't modify the binning from here: act on maxEtaCut and PtThresholdForMaxEtaCut instead
     {"hcountD0D0barPerEvent", "D0,D0bar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 10.}}}}}};

  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    int counterD0D0bar = 0;
    registry.fill(HIST("hMCEvtCount"), 0);
    //MC gen level
    for (auto& particle1 : particlesMC) {
      if (cutEtaCandMax >= 0. && std::abs(particle1.eta()) > cutEtaCandMax)
        continue;
      if (cutPtCandMin >= 0. && std::abs(particle1.pt()) < cutPtCandMin)
        continue;
      //just checking if the particle is D0 or D0bar, for now
      if (std::abs(particle1.pdgCode()) == 421) {
        registry.fill(HIST("hptcandMCGen"), particle1.pt());
        registry.fill(HIST("hEtaMCGen"), particle1.eta());
        registry.fill(HIST("hPhiMCGen"), particle1.phi());
        registry.fill(HIST("hYMCGen"), RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode())));
        counterD0D0bar++;

        //D-Dbar correlation dedicated section
        //if it's a D0 particle, search for D0bar and evaluate correlations
        if (particle1.pdgCode() == 421) {                  //just checking the particle PDG, not the decay channel (differently from Reco: you have a BR factor btw such levels!)
          registry.fill(HIST("hcountD0triggersMCGen"), 0); //to count trigger D0 (for normalisation)
          for (auto& particle2 : particlesMC) {
            if (cutEtaCandMax >= 0. && std::abs(particle2.eta()) > cutEtaCandMax)
              continue;
            if (cutPtCandMin >= 0. && std::abs(particle2.pt()) < cutPtCandMin)
              continue;
            if (particle2.pdgCode() == -421) {
              double eta1 = particle1.eta(), eta2 = particle2.eta(), pt1 = particle1.pt(), pt2 = particle2.pt(), phi1 = particle1.phi(), phi2 = particle2.phi();
              registry.fill(HIST("hDeltaEtaPtIntMCGen"), eta2 - eta1);
              registry.fill(HIST("hDeltaPhiPtIntMCGen"), getDeltaPhi(phi2, phi1));
              registry.fill(HIST("hCorrel2DPtIntMCGen"), getDeltaPhi(phi2, phi1), eta2 - eta1);
              registry.fill(HIST("hDeltaEtaVsPtMCGen"), pt1, pt2, eta2 - eta1);
              registry.fill(HIST("hDeltaPhiVsPtMCGen"), pt1, pt2, getDeltaPhi(phi2, phi1));
              registry.fill(HIST("hDeltaPtDDbarMCGen"), pt2 - pt1);
              registry.fill(HIST("hDeltaPtMaxMinMCGen"), std::max(pt2, pt1) - std::min(pt2, pt1));
              double etaCut = 0.;
              double ptCut = 0.;
              do { //fill pairs vs etaCut plot
                ptCut = 0.;
                etaCut += 0.1;
                do { //fill pairs vs etaCut plot
                  if (std::abs(particle1.eta()) < etaCut && std::abs(particle2.eta()) < etaCut && particle1.pt() > ptCut && particle2.pt() > ptCut)
                    registry.fill(HIST("hDDbarVsEtaCutMCGen"), etaCut - 0.01, ptCut + 0.01);
                  ptCut += 0.5;
                } while (ptCut < PtThresholdForMaxEtaCut - 0.05);
              } while (etaCut < maxEtaCut - 0.05);
            } // end D0bar check
          }   //end inner loop
        }     //end D0 check
      }       //end outer if (MC check D0/D0bar)

    } //end outer loop
    registry.fill(HIST("hcountD0D0barPerEvent"), counterD0D0bar);
  }
};

/// D0 analysis task - LIKE SIGN - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
struct TaskD0D0barCorrelationLS {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hmassD0 for normalisation, and hmass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hmass", "D0,D0bar candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{200, 1., 3.}}}},
     {"hmassD0", "D0,D0bar candidates;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{200, 1., 3.}}}},
     {"hmassD0bar", "D0,D0bar candidates;inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{200, 1., 3.}}}},
     {"hmass2DCorrelationPairs", "D0,D0bar candidates 2D;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{200, 1., 3.}, {200, 1., 3.}}}},
     {"hptcand", "D0,D0bar candidates;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0", "D0,D0bar candidates;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1", "D0,D0bar candidates;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hselectionstatus", "D0,D0bar candidates;selection status;entries", {HistType::kTH1F, {{5, -0.5, 4.5}}}},
     {"hEta", "D0,D0bar candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhi", "D0,D0bar candidates;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hY", "D0,D0bar candidates;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDeltaEtaPtInt", "D0,D0bar candidates;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtInt", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtInt", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaEtaVsPt", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {200, -10., 10.}}}},
     {"hDeltaPhiVsPt", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPtDDbar", "D0,D0bar candidates;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMin", "D0,D0bar candidates;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<aod::Big2Prong> const& candidates, aod::BigTracks const& tracks)
  {
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutEtaCandMax >= 0. && std::abs(candidate1.eta()) > cutEtaCandMax)
        continue;
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin)
        continue;
      //fill invariant mass plots and generic info from all D0/D0bar candidates
      if (candidate1.isSelD0() >= d_selectionFlagD0) {
        registry.fill(HIST("hmass"), InvMassD0(candidate1));
        registry.fill(HIST("hmassD0"), InvMassD0(candidate1));
      }
      if (candidate1.isSelD0bar() >= d_selectionFlagD0bar) {
        registry.fill(HIST("hmass"), InvMassD0bar(candidate1));
        registry.fill(HIST("hmassD0bar"), InvMassD0bar(candidate1));
      }
      registry.fill(HIST("hptcand"), candidate1.pt());
      registry.fill(HIST("hptprong0"), candidate1.ptProng0());
      registry.fill(HIST("hptprong1"), candidate1.ptProng1());
      registry.fill(HIST("hEta"), candidate1.eta());
      registry.fill(HIST("hPhi"), candidate1.phi());
      registry.fill(HIST("hY"), YD0(candidate1));
      registry.fill(HIST("hselectionstatus"), candidate1.isSelD0() + (candidate1.isSelD0bar() * 2));

      double ptParticle1 = candidate1.pt(); //trigger particle is the largest-pT one

      //D-Dbar correlation dedicated section
      //For like-sign, first loop on both D0 and D0bars. First candidate is for sure a D0 and D0bars (checked before, so don't re-check anything on it)
      for (auto& candidate2 : candidates) {
        //check decay channel flag for candidate2
        if (!(candidate2.hfflag() & 1 << D0ToPiK)) {
          continue;
        }
        //for the associated, has to have smaller pT, and pass D0sel if trigger passes D0sel, or D0barsel if trigger passes D0barsel
        if (candidate2.pt() < ptParticle1 && ((candidate1.isSelD0() >= d_selectionFlagD0 && candidate2.isSelD0() >= d_selectionFlagD0) || (candidate1.isSelD0bar() >= d_selectionFlagD0bar && candidate2.isSelD0bar() >= d_selectionFlagD0bar))) {
          if (cutEtaCandMax >= 0. && std::abs(candidate2.eta()) > cutEtaCandMax)
            continue;
          if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin)
            continue;
          //Excluding self-correlations (in principle not possible due to the '<' condition, but could rounding break it?)
          if (candidate1.mRowIndex == candidate2.mRowIndex)
            continue;
          double eta1 = candidate1.eta(), eta2 = candidate2.eta(), pt1 = candidate1.pt(), pt2 = candidate2.pt(), phi1 = candidate1.phi(), phi2 = candidate2.phi();
          registry.fill(HIST("hmass2DCorrelationPairs"), InvMassD0(candidate1), InvMassD0bar(candidate2));
          registry.fill(HIST("hDeltaEtaPtInt"), eta2 - eta1);
          registry.fill(HIST("hDeltaPhiPtInt"), getDeltaPhi(phi2, phi1));
          registry.fill(HIST("hCorrel2DPtInt"), getDeltaPhi(phi2, phi1), eta2 - eta1);
          registry.fill(HIST("hDeltaEtaVsPt"), pt1, pt2, eta2 - eta1);
          registry.fill(HIST("hDeltaPhiVsPt"), pt1, pt2, getDeltaPhi(phi2, phi1));
          registry.fill(HIST("hDeltaPtDDbar"), pt2 - pt1);
          registry.fill(HIST("hDeltaPtMaxMin"), std::max(pt2, pt1) - std::min(pt2, pt1));
        }
        //note: candidates selected as both D0 and D0bar are used, and considered in both situation (but not auto-correlated): reflections could play a relevant role.
        //another, more restrictive, option, could be to consider only candidates selected with a single option (D0 xor D0bar)

      } // end inner loop (Dbars)

    } //end outer loop
  }
};

/// D0 analysis task - LIKE SIGN - for MC reco analysis (data-like but matching to true DO and D0bar)
struct TaskD0D0barCorrelationMCRecLS {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hmassD0MCRec for normalisation, and hmass2DCorrelationPairsMCRec for 2D-sideband-subtraction purposes
    {{"hmassD0MCRec", "D0,D0bar candidates - MC reco;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmassD0barMCRec", "D0,D0bar candidates - MC reco;inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{500, 0., 5.}}}},
     {"hmass2DCorrelationPairsMCRec", "D0,D0bar candidates 2D;inv. mass D0 only (#pi K) (GeV/#it{c}^{2});inv. mass D0bar only (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH2F, {{200, 1., 3.}, {200, 1., 3.}}}},
     {"hptcandMCRec", "D0,D0bar candidates - MC reco;candidate #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong0MCRec", "D0,D0bar candidates - MC reco;prong 0 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hptprong1MCRec", "D0,D0bar candidates - MC reco;prong 1 #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hEtaMCRec", "D0,D0bar candidates - MC reco;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCRec", "D0,D0bar candidates - MC reco;candidate #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCRec", "D0,D0bar candidates - MC reco;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDeltaEtaPtIntMCRec", "D0,D0bar candidates - MC reco;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCRec", "D0,D0bar candidates - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCRec", "D0,D0bar candidates - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaEtaVsPtMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {200, -10., 10.}}}},
     {"hDeltaPhiVsPtMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPtDDbarMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCRec", "D0,D0bar candidates - MC reco;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<aod::Big2ProngMC> const& candidates, aod::BigTracks const& tracks)
  {
    //MC reco level
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutEtaCandMax >= 0. && std::abs(candidate1.eta()) > cutEtaCandMax)
        continue;
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin)
        continue;
      if (std::abs(candidate1.flagMCMatchRec()) == 1 << D0ToPiK) {
        //fill invariant mass plots and generic info from all D0/D0bar candidates
        if (candidate1.isSelD0() >= d_selectionFlagD0 && candidate1.flagMCMatchRec() == D0ToPiK) { //only reco and matched as D0
          registry.fill(HIST("hmassD0MCRec"), InvMassD0(candidate1));
        }
        if (candidate1.isSelD0bar() >= d_selectionFlagD0bar && candidate1.flagMCMatchRec() == D0ToPiK) { //only reco and matched as D0bar
          registry.fill(HIST("hmassD0barMCRec"), InvMassD0bar(candidate1));
        }
        registry.fill(HIST("hptcandMCRec"), candidate1.pt());
        registry.fill(HIST("hptprong0MCRec"), candidate1.ptProng0());
        registry.fill(HIST("hptprong1MCRec"), candidate1.ptProng1());
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
          bool conditionLSForD0 = (candidate1.isSelD0() >= d_selectionFlagD0bar && candidate1.flagMCMatchRec() == 1 << D0ToPiK) && (candidate2.isSelD0() >= d_selectionFlagD0bar && candidate2.flagMCMatchRec() == 1 << D0ToPiK);
          bool conditionLSForD0bar = (candidate1.isSelD0bar() >= d_selectionFlagD0bar && candidate1.flagMCMatchRec() == -1 << D0ToPiK) && (candidate2.isSelD0bar() >= d_selectionFlagD0bar && candidate2.flagMCMatchRec() == -1 << D0ToPiK);
          if (candidate2.pt() < ptParticle1 && (conditionLSForD0 || conditionLSForD0bar)) { //LS pair (of D0 or of D0bar) + pt2<pt1
            if (cutEtaCandMax >= 0. && std::abs(candidate2.eta()) > cutEtaCandMax)
              continue;
            if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin)
              continue;
            //Excluding self-correlations (in principle not possible due to the '<' condition, but could rounding break it?)
            if (candidate1.mRowIndex == candidate2.mRowIndex)
              continue;
            double eta1 = candidate1.eta(), eta2 = candidate2.eta(), pt1 = candidate1.pt(), pt2 = candidate2.pt(), phi1 = candidate1.phi(), phi2 = candidate2.phi();
            registry.fill(HIST("hmass2DCorrelationPairsMCRec"), InvMassD0(candidate1), InvMassD0bar(candidate2));
            registry.fill(HIST("hDeltaEtaPtIntMCRec"), eta2 - eta1);
            registry.fill(HIST("hDeltaPhiPtIntMCRec"), getDeltaPhi(phi2, phi1));
            registry.fill(HIST("hCorrel2DPtIntMCRec"), getDeltaPhi(phi2, phi1), eta2 - eta1);
            registry.fill(HIST("hDeltaEtaVsPtMCRec"), pt1, pt2, eta2 - eta1);
            registry.fill(HIST("hDeltaPhiVsPtMCRec"), pt1, pt2, getDeltaPhi(phi2, phi1));
            registry.fill(HIST("hDeltaPtDDbarMCRec"), pt2 - pt1);
            registry.fill(HIST("hDeltaPtMaxMinMCRec"), std::max(pt2, pt1) - std::min(pt2, pt1));
          } //end inner if (MC match)

        } // end inner loop (Dbars)
      }   //end outer if (MC match)
    }     //end outer loop
  }
};

/// D0 analysis task - for MC gen-level analysis, like sign particles
struct TaskD0D0barCorrelationMCGenLS {

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hptcandMCGen", "D0,D0bar LS particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hcountD0triggersMCGen", "D0,D0bar LS trigger particles (to be divided by two) - MC gen;;N of triggers", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hEtaMCGen", "D0,D0bar LS particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "D0,D0bar LS particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hYMCGen", "D0,D0bar candidates - MC gen;candidate #it{#y};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hDeltaEtaPtIntMCGen", "D0,D0bar LS particles - MC gen;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCGen", "D0,D0bar LS particles - MC gen;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCGen", "D0,D0bar LS particles - MC gen;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaEtaVsPtMCGen", "D0,D0bar LS LS candidates - MC gen;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {200, -10., 10.}}}},
     {"hDeltaPhiVsPtMCGen", "D0,D0bar LS candidates - MC gen;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPtDDbarMCGen", "D0,D0bar LS particles - MC gen;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCGen", "D0,D0bar LS particles - MC gen;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hcountD0D0barPerEvent", "D0,D0bar LS particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}}}};

  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    int counterD0D0bar = 0;
    registry.fill(HIST("hMCEvtCount"), 0);
    //MC gen level
    for (auto& particle1 : particlesMC) {
      if (cutEtaCandMax >= 0. && std::abs(particle1.eta()) > cutEtaCandMax)
        continue;
      if (cutPtCandMin >= 0. && std::abs(particle1.pt()) < cutPtCandMin)
        continue;

      double ptParticle1 = particle1.pt(); //trigger particle is the largest pT one

      //Check whether particle is D0 or D0bar (and not the decay chain)
      if (std::abs(particle1.pdgCode()) == 421) {
        registry.fill(HIST("hptcandMCGen"), particle1.pt());
        registry.fill(HIST("hEtaMCGen"), particle1.eta());
        registry.fill(HIST("hPhiMCGen"), particle1.phi());
        registry.fill(HIST("hYMCGen"), RecoDecay::Y(array{particle1.px(), particle1.py(), particle1.pz()}, RecoDecay::getMassPDG(particle1.pdgCode())));
        counterD0D0bar++;
        //D-Dbar correlation dedicated section
        //if it's D0, search for D0bar and evaluate correlations.
        registry.fill(HIST("hcountD0triggersMCGen"), 0); //to count trigger D0 (normalisation)
        for (auto& particle2 : particlesMC) {
          if (cutEtaCandMax >= 0. && std::abs(particle2.eta()) > cutEtaCandMax)
            continue;
          if (cutPtCandMin >= 0. && std::abs(particle2.pt()) < cutPtCandMin)
            continue;
          if (particle2.pt() < ptParticle1 && particle2.pdgCode() == particle1.pdgCode()) { //like-sign condition (both 421 or both -421) and pT_Trig>pT_assoc
            //Excluding self-correlations (in principle not possible due to the '<' condition, but could rounding break it?)
            if (particle1.mRowIndex == particle2.mRowIndex)
              continue;
            double eta1 = particle1.eta(), eta2 = particle2.eta(), pt1 = particle1.pt(), pt2 = particle2.pt(), phi1 = particle1.phi(), phi2 = particle2.phi();
            registry.fill(HIST("hDeltaEtaPtIntMCGen"), eta2 - eta1);
            registry.fill(HIST("hDeltaPhiPtIntMCGen"), getDeltaPhi(phi2, phi1));
            registry.fill(HIST("hCorrel2DPtIntMCGen"), getDeltaPhi(phi2, phi1), eta2 - eta1);
            registry.fill(HIST("hDeltaEtaVsPtMCGen"), pt1, pt2, eta2 - eta1);
            registry.fill(HIST("hDeltaPhiVsPtMCGen"), pt1, pt2, getDeltaPhi(phi2, phi1));
            registry.fill(HIST("hDeltaPtDDbarMCGen"), pt2 - pt1);
            registry.fill(HIST("hDeltaPtMaxMinMCGen"), std::max(pt2, pt1) - std::min(pt2, pt1));
          }
        } // end inner loop (Dbars)
      }   //end outer if (MC check D0)
    }     //end outer loop
    registry.fill(HIST("hcountD0D0barPerEvent"), counterD0D0bar);
  }
};

/// c-cbar correlation task analysis task - for MC gen-level analysis
struct TaskCCbarCorrelationMCGen {

  HistogramRegistry registry{
    "registry",
    {{"hMCEvtCount", "Event counter - MC gen;;entries", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hptcandMCGen", "c,cbar particles - MC gen;particle #it{p}_{T} (GeV/#it{c});entries", {HistType::kTH1F, {{100, 0., 10.}}}},
     {"hcountD0triggersMCGen", "c trigger particles - MC gen;;N of trigger c", {HistType::kTH1F, {{1, -0.5, 0.5}}}},
     {"hEtaMCGen", "c,cbar particles - MC gen;particle #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiMCGen", "c,cbar particles - MC gen;particle #it{#varphi};entries", {HistType::kTH1F, {{32, 0., 2. * o2::constants::math::PI}}}},
     {"hDeltaEtaPtIntMCGen", "c,cbar particles - MC gen;#it{#eta}^{cbar}-#it{#eta}^{c};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCGen", "c,cbar particles - MC gen;#it{#varphi}^{cbar}-#it{#varphi}^{c};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCGen", "c,cbar particles - MC gen;#it{#varphi}^{cbar}-#it{#varphi}^{c};#it{#eta}^{cbar}-#it{#eta}^{c};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaEtaVsPtMCGen", "c,cbar candidates - MC gen;#it{p}_{T}^{c};#it{p}_{T}^{c}bar};#it{#eta}^{D0bar}-#it{#eta}^{c};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {200, -10., 10.}}}},
     {"hDeltaPhiVsPtMCGen", "c,cbar candidates - MC gen;#it{p}_{T}^{c};#it{p}_{T}^{cbar};#it{#varphi}^{D0bar}-#it{#varphi}^{c};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPtDDbarMCGen", "c,cbar particles - MC gen;#it{p}_{T}^{cbar}-#it{p}_{T}^{c};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCGen", "c,cbar particles - MC gen;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hcountCCbarPerEvent", "c,cbar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}},
     {"hcountCCbarPerEventPreEtaCut", "c,cbar particles - MC gen;Number per event;entries", {HistType::kTH1F, {{20, 0., 20.}}}}}};

  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  void process(aod::McCollision const& mccollision, soa::Join<aod::McParticles, aod::HfCandProng2MCGen> const& particlesMC)
  {
    registry.fill(HIST("hMCEvtCount"), 0);
    int counterccbar = 0, counterccbarPreEtasel = 0;

    //loop over particles at MC gen level
    for (auto& particle1 : particlesMC) {
      if (std::abs(particle1.pdgCode()) == 4) { //c or cbar quark found
        int partMothPDG = particlesMC.iteratorAt(particle1.mother0()).pdgCode();
        if (std::abs(partMothPDG) != 4)
          counterccbarPreEtasel++; //count c or cbar when it doesn't come from itself via fragmenatation (c->c+X)
        if (cutEtaCandMax >= 0. && std::abs(particle1.eta()) > cutEtaCandMax)
          continue;
        if (cutPtCandMin >= 0. && std::abs(particle1.pt()) < cutPtCandMin)
          continue;
        registry.fill(HIST("hptcandMCGen"), particle1.pt());
        registry.fill(HIST("hEtaMCGen"), particle1.eta());
        registry.fill(HIST("hPhiMCGen"), particle1.phi());
        if (std::abs(partMothPDG) != 4)
          counterccbar++; //count if c or cbar don't come from themselves during fragmenatation (after kinematic selection)

        //c-cbar correlation dedicated section
        //if it's c, search for cbar and evaluate correlations.
        if (particle1.pdgCode() == 4) {
          //check whether mothers of quark c are still '4' particles - in that case the c quark comes from its own fragmentation, skip it
          if (partMothPDG == 4)
            continue;
          registry.fill(HIST("hcountD0triggersMCGen"), 0); //to count trigger c quark (for normalisation)

          for (auto& particle2 : particlesMC) {
            if (cutEtaCandMax >= 0. && std::abs(particle2.eta()) > cutEtaCandMax)
              continue;
            if (cutPtCandMin >= 0. && std::abs(particle2.pt()) < cutPtCandMin)
              continue;
            if (particle2.pdgCode() == -4) {
              //check whether mothers of quark cbar are still '-4' particles - in that case the cbar quark comes from its own fragmentation, skip it
              if (particlesMC.iteratorAt(particle2.mother0()).pdgCode() == -4)
                continue;
              double eta1 = particle1.eta(), eta2 = particle2.eta(), pt1 = particle1.pt(), pt2 = particle2.pt(), phi1 = particle1.phi(), phi2 = particle2.phi();
              registry.fill(HIST("hDeltaEtaPtIntMCGen"), eta2 - eta1);
              registry.fill(HIST("hDeltaPhiPtIntMCGen"), getDeltaPhi(phi2, phi1));
              registry.fill(HIST("hCorrel2DPtIntMCGen"), getDeltaPhi(phi2, phi1), eta2 - eta1);
              registry.fill(HIST("hDeltaEtaVsPtMCGen"), pt1, pt2, eta2 - eta1);
              registry.fill(HIST("hDeltaPhiVsPtMCGen"), pt1, pt2, getDeltaPhi(phi2, phi1));
              registry.fill(HIST("hDeltaPtDDbarMCGen"), pt2 - pt1);
              registry.fill(HIST("hDeltaPtMaxMinMCGen"), std::max(particle2.pt(), particle1.pt()) - std::min(particle2.pt(), particle1.pt()));
            } // end outer if (check cbar)
          }   // end inner loop
        }     //end outer if (check c)
      }       //end c/cbar if
    }         //end outer loop
    registry.fill(HIST("hcountCCbarPerEvent"), counterccbar);
    registry.fill(HIST("hcountCCbarPerEventPreEtaCut"), counterccbarPreEtasel);
  }
};

/// checks phi resolution for standard definition and sec-vtx based definition
struct TaskD0D0barCorrelationCheckPhiResolution {

  HistogramRegistry registry{
    "registry",
    {{"hmass", "D0,D0bar candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{200, 1., 3.}}}},
     {"hEta", "D0,D0bar candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiStdPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, 0., 2. * o2::constants::math::PI}, {50, 0., 50.}}}},
     {"hPhiByVtxPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, 0., 2. * o2::constants::math::PI}, {50, 0., 50.}}}},
     {"hPhiDifferenceTwoMethods", "D0,D0bar candidates;candidate #it{#Delta#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {50, 0., 50.}}}},
     {"hDifferenceGenPhiStdPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {50, 0., 50.}}}},
     {"hDifferenceGenPhiByVtxPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {50, 0., 50.}}}},
     {"hDeltaPhiPtIntStdPhi", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPhiPtIntByVtxPhi", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPhiVsPtStdPhi", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPhiVsPtByVtxPhi", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{100, 0., 10.}, {100, 0., 10.}, {128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}}}};

  Configurable<int> d_selectionFlagD0{"d_selectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> d_selectionFlagD0bar{"d_selectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutEtaCandMax{"cutEtaCandMax", -1., "max. cand. pseudorapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= d_selectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= d_selectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<aod::Big2ProngMC> const& candidates, aod::McParticles const& particlesMC, aod::BigTracksMC const& tracksMC)
  {
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutEtaCandMax >= 0. && std::abs(candidate1.eta()) > cutEtaCandMax)
        continue;
      if (cutPtCandMin >= 0. && std::abs(candidate1.pt()) < cutPtCandMin)
        continue;
      registry.fill(HIST("hmass"), InvMassD0(candidate1));
      registry.fill(HIST("hEta"), candidate1.eta());

      //D-Dbar correlation dedicated section
      //if it's a candidate D0, search for D0bar and evaluate correlations
      if (candidate1.isSelD0() >= d_selectionFlagD0) {
        double xPrimaryVertex = candidate1.index0_as<aod::BigTracksMC>().collision().posX(), yPrimaryVertex = candidate1.index0_as<aod::BigTracksMC>().collision().posY();
        double pt1 = candidate1.pt(), phi1Std = candidate1.phi();
        double phi1ByVtx = evaluatePhiByVertex(xPrimaryVertex, candidate1.xSecondaryVertex(), yPrimaryVertex, candidate1.ySecondaryVertex());
        registry.fill(HIST("hPhiStdPhi"), phi1Std, pt1);
        registry.fill(HIST("hPhiByVtxPhi"), phi1ByVtx, pt1); //trick to have correct Phi range
        registry.fill(HIST("hPhiDifferenceTwoMethods"), getDeltaPhi(phi1ByVtx, phi1Std), pt1);

        //get corresponding gen-level D0, if exists, and evaluate gen-rec phi-difference with two approaches
        if (std::abs(candidate1.flagMCMatchRec()) == 1 << D0ToPiK) {                                                     //ok to keep both D0 and D0bar
          int indexGen = RecoDecay::getMother(particlesMC, candidate1.index0_as<aod::BigTracksMC>().label(), 421, true); //MC-gen corresponding index for MC-reco candidate
          if (indexGen > 0) {
            double phi1Gen = particlesMC.iteratorAt(indexGen).phi();
            registry.fill(HIST("hDifferenceGenPhiStdPhi"), getDeltaPhi(phi1Std, phi1Gen), pt1);
            registry.fill(HIST("hDifferenceGenPhiByVtxPhi"), getDeltaPhi(phi1ByVtx, phi1Gen), pt1);
          }
        }

        for (auto& candidate2 : candidates) {
          //check decay channel flag for candidate2
          if (!(candidate2.hfflag() & 1 << D0ToPiK)) {
            continue;
          }
          if (candidate2.isSelD0bar() >= d_selectionFlagD0bar) { //accept only D0bar candidates
            if (cutEtaCandMax >= 0. && std::abs(candidate2.eta()) > cutEtaCandMax)
              continue;
            if (cutPtCandMin >= 0. && std::abs(candidate2.pt()) < cutPtCandMin)
              continue;
            //Excluding self-correlations (could happen in case of reflections)
            if (candidate1.mRowIndex == candidate2.mRowIndex)
              continue;
            double pt2 = candidate2.pt(), phi2Std = candidate2.phi();
            double phi2ByVtx = evaluatePhiByVertex(xPrimaryVertex, candidate2.xSecondaryVertex(), yPrimaryVertex, candidate2.ySecondaryVertex());
            registry.fill(HIST("hDeltaPhiPtIntStdPhi"), getDeltaPhi(phi2Std, phi1Std));
            registry.fill(HIST("hDeltaPhiPtIntByVtxPhi"), getDeltaPhi(phi2ByVtx, phi1ByVtx));
            registry.fill(HIST("hDeltaPhiVsPtStdPhi"), pt1, pt2, getDeltaPhi(phi2Std, phi1Std));
            registry.fill(HIST("hDeltaPhiVsPtByVtxPhi"), pt1, pt2, getDeltaPhi(phi2ByVtx, phi1ByVtx));
          }
        } // end inner loop (Dbars)
      }
    } //end outer loop
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<CreateBig2Prong>("add-collision-id"),
    adaptAnalysisTask<TaskD0D0barCorrelation>("hf-task-d0d0bar-correlation"),
    adaptAnalysisTask<TaskD0D0barCorrelationLS>("hf-task-d0d0bar-correlation-ls")};
  //MC-based tasks
  const bool doMC = cfgc.options().get<bool>("doMC");
  if (doMC) {
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationMCRec>("hf-task-d0d0bar-correlation-mc-rec"));
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationMCGen>("hf-task-d0d0bar-correlation-mc-gen"));
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationMCRecLS>("hf-task-d0d0bar-correlation-mc-rec-ls"));
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationMCGenLS>("hf-task-d0d0bar-correlation-mc-gen-ls"));
    workflow.push_back(adaptAnalysisTask<TaskCCbarCorrelationMCGen>("hf-task-ccbar-correlation-mc-gen"));
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationCheckPhiResolution>("hf-task-d0d0bar-correlation-crosscheck-phi"));
  }
  return workflow;
}
