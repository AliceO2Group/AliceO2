// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskCorrelationDDbar.cxx
/// \brief D-Dbar analysis task - data-like, MC-reco and MC-kine analyses. For ULS and LS pairs
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

namespace o2::aod
{
using DDbarPairFull = soa::Join<aod::DDbarPair, aod::DDbarRecoInfo>;
} // namespace o2::aod

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMCGen{"doMCGen", VariantType::Bool, false, {"Run MC-Gen dedicated tasks."}};
  ConfigParamSpec optionDoMCRec{"doMCRec", VariantType::Bool, true, {"Run MC-Rec dedicated tasks."}};
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

///
/// Returns phi of candidate/particle evaluated from x and y components of segment connecting primary and secondary vertices
///
double evaluatePhiByVertex(double xVertex1, double xVertex2, double yVertex1, double yVertex2)
{
  return RecoDecay::Phi(xVertex2 - xVertex1, yVertex2 - yVertex1);
}

// string definitions, used for histogram axis labels
const TString stringPtD = "#it{p}_{T}^{D} (GeV/#it{c});";
const TString stringPtDbar = "#it{p}_{T}^{Dbar} (GeV/#it{c});";
const TString stringDeltaPt = "#it{p}_{T}^{Dbar}-#it{p}_{T}^{D} (GeV/#it{c});";
const TString stringDeltaPtMaxMin = "#it{p}_{T}^{max}-#it{p}_{T}^{min} (GeV/#it{c});";
const TString stringDeltaEta = "#it{#eta}^{Dbar}-#it{#eta}^{D};";
const TString stringDeltaPhi = "#it{#varphi}^{Dbar}-#it{#varphi}^{D} (rad);";
const TString stringDDbar = "D,Dbar candidates ";
const TString stringSignal = "signal region;";
const TString stringSideband = "sidebands;";
const TString stringMCParticles = "MC gen - D,Dbar particles;";
const TString stringMCReco = "MC reco - D,Dbar candidates ";

//definition of vectors for standard ptbin and invariant mass configurables
const int npTBinsCorrelations = 8;
const double pTBinsCorrelations[npTBinsCorrelations + 1] = {0., 2., 4., 6., 8., 12., 16., 24., 99.};
auto pTBinsCorrelations_v = std::vector<double>{pTBinsCorrelations, pTBinsCorrelations + npTBinsCorrelations + 1};
const double signalRegionInnerDefault[npTBinsCorrelations + 1] = {1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75};
const double signalRegionOuterDefault[npTBinsCorrelations + 1] = {1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81, 1.81};
const double sidebandLeftInnerDefault[npTBinsCorrelations + 1] = {1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84, 1.84};
const double sidebandLeftOuterDefault[npTBinsCorrelations + 1] = {1.90, 1.90, 1.90, 1.90, 1.90, 1.90, 1.90, 1.90};
const double sidebandRightInnerDefault[npTBinsCorrelations + 1] = {1.93, 1.93, 1.93, 1.93, 1.93, 1.93, 1.93, 1.93};
const double sidebandRightOuterDefault[npTBinsCorrelations + 1] = {1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99, 1.99};
auto signalRegionInner_v = std::vector<double>{signalRegionInnerDefault, signalRegionInnerDefault + npTBinsCorrelations};
auto signalRegionOuter_v = std::vector<double>{signalRegionOuterDefault, signalRegionOuterDefault + npTBinsCorrelations};
auto sidebandLeftInner_v = std::vector<double>{sidebandLeftInnerDefault, sidebandLeftInnerDefault + npTBinsCorrelations};
auto sidebandLeftOuter_v = std::vector<double>{sidebandLeftOuterDefault, sidebandLeftOuterDefault + npTBinsCorrelations};
auto sidebandRightInner_v = std::vector<double>{sidebandRightInnerDefault, sidebandRightInnerDefault + npTBinsCorrelations};
auto sidebandRightOuter_v = std::vector<double>{sidebandRightOuterDefault, sidebandRightOuterDefault + npTBinsCorrelations};

/// D-Dbar correlation pair filling task, from pair tables - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
/// Works on both USL and LS analyses pair tables
struct HfTaskCorrelationDDbar {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 (from correlator task) for normalisation, and hMass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hMass2DCorrelationPairs", stringDDbar + "2D;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});" + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSignalRegion", stringDDbar + stringSignal + stringDeltaEta + "entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSignalRegion", stringDDbar + stringSignal + stringDeltaPhi + "entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSignalRegion", stringDDbar + stringSignal + stringDeltaPhi + stringDeltaEta + "entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSignalRegion", stringDDbar + stringSignal + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarSignalRegion", stringDDbar + stringSignal + stringDeltaPt + "entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSignalRegion", stringDDbar + stringSignal + stringDeltaPtMaxMin + "entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hDeltaEtaPtIntSidebands", stringDDbar + stringSideband + stringDeltaEta + "entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSidebands", stringDDbar + stringSideband + stringDeltaPhi + "entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSidebands", stringDDbar + stringSideband + stringDeltaPhi + stringDeltaEta + "entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSidebands", stringDDbar + stringSideband + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarSidebands", stringDDbar + stringSideband + stringDeltaPt + "entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSidebands", stringDDbar + stringSideband + stringDeltaPtMaxMin + "entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{pTBinsCorrelations_v}, "pT bin limits for correlation plots"};
  //signal and sideband region edges, to be defined via json file (initialised to empty)
  Configurable<std::vector<double>> signalRegionInner{"signalRegionInner", std::vector<double>{signalRegionInner_v}, "Inner values of signal region vs pT"};
  Configurable<std::vector<double>> signalRegionOuter{"signalRegionOuter", std::vector<double>{signalRegionOuter_v}, "Outer values of signal region vs pT"};
  Configurable<std::vector<double>> sidebandLeftInner{"sidebandLeftInner", std::vector<double>{sidebandLeftInner_v}, "Inner values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandLeftOuter{"sidebandLeftOuter", std::vector<double>{sidebandLeftOuter_v}, "Outer values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightInner{"sidebandRightInner", std::vector<double>{sidebandRightInner_v}, "Inner values of right sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightOuter{"sidebandRightOuter", std::vector<double>{sidebandRightOuter_v}, "Outer values of right sideband vs pT"};

  void init(o2::framework::InitContext&)
  {
    // redefinition of pT axes for THnSparse holding correlation entries
    int nBinspTaxis = binsCorrelations->size() - 1;
    const double* valuespTaxis = binsCorrelations->data();

    for (int i = 2; i <= 3; i++) {
      registry.get<THnSparse>(HIST("hMass2DCorrelationPairs"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSignalRegion"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSidebands"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
    }
  }

  void process(aod::DDbarPairFull const& pairEntries)
  {
    for (auto& pairEntry : pairEntries) {
      //define variables for widely used quantities
      double deltaPhi = pairEntry.deltaPhi();
      double deltaEta = pairEntry.deltaEta();
      double ptD = pairEntry.ptD();
      double ptDbar = pairEntry.ptDbar();
      double massD = pairEntry.mD();
      double massDbar = pairEntry.mDbar();

      //fill 2D invariant mass plots
      registry.fill(HIST("hMass2DCorrelationPairs"), massD, massDbar, ptD, ptDbar);

      //reject entries outside pT ranges of interest
      int pTBinD = o2::analysis::findBin(binsCorrelations, ptD);
      int pTBinDbar = o2::analysis::findBin(binsCorrelations, ptDbar);
      if (pTBinD == -1 || pTBinDbar == -1) { //at least one particle outside accepted pT range
        continue;
      }

      //check if correlation entry belongs to signal region, sidebands or is outside both, and fill correlation plots
      if (massD > signalRegionInner->at(pTBinD) && massD < signalRegionOuter->at(pTBinD) && massDbar > signalRegionInner->at(pTBinDbar) && massDbar < signalRegionOuter->at(pTBinDbar)) {
        //in signal region
        registry.fill(HIST("hCorrel2DVsPtSignalRegion"), deltaPhi, deltaEta, ptD, ptDbar);
        registry.fill(HIST("hCorrel2DPtIntSignalRegion"), deltaPhi, deltaEta);
        registry.fill(HIST("hDeltaEtaPtIntSignalRegion"), deltaEta);
        registry.fill(HIST("hDeltaPhiPtIntSignalRegion"), deltaPhi);
        registry.fill(HIST("hDeltaPtDDbarSignalRegion"), ptDbar - ptD);
        registry.fill(HIST("hDeltaPtMaxMinSignalRegion"), std::abs(ptDbar - ptD));
      }

      if ((massD > sidebandLeftInner->at(pTBinD) && massD < sidebandLeftOuter->at(pTBinD) && massDbar > sidebandLeftInner->at(pTBinDbar) && massDbar < sidebandRightOuter->at(pTBinDbar)) ||
          (massD > sidebandRightInner->at(pTBinD) && massD < sidebandRightOuter->at(pTBinD) && massDbar > sidebandLeftInner->at(pTBinDbar) && massDbar < sidebandRightOuter->at(pTBinDbar)) ||
          (massD > sidebandLeftInner->at(pTBinD) && massD < sidebandRightOuter->at(pTBinD) && massDbar > sidebandLeftInner->at(pTBinDbar) && massDbar < sidebandLeftOuter->at(pTBinDbar)) ||
          (massD > sidebandLeftInner->at(pTBinD) && massD < sidebandRightOuter->at(pTBinD) && massDbar > sidebandRightInner->at(pTBinDbar) && massDbar < sidebandRightOuter->at(pTBinDbar))) {
        //in sideband region
        registry.fill(HIST("hCorrel2DVsPtSidebands"), deltaPhi, deltaEta, ptD, ptDbar);
        registry.fill(HIST("hCorrel2DPtIntSidebands"), deltaPhi, deltaEta);
        registry.fill(HIST("hDeltaEtaPtIntSidebands"), deltaEta);
        registry.fill(HIST("hDeltaPhiPtIntSidebands"), deltaPhi);
        registry.fill(HIST("hDeltaPtDDbarSidebands"), ptDbar - ptD);
        registry.fill(HIST("hDeltaPtMaxMinSidebands"), std::abs(ptDbar - ptD));
      }
    } //end loop
  }
};

/// D-Dbar correlation pair filling task, from pair tables - for MC reco-level analysis (candidates matched to true signal only, but also bkg sources are studied)
/// Works on both USL and LS analyses pair tables
struct HfTaskCorrelationDDbarMcRec {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 (from correlator task) for normalisation, and hMass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hMass2DCorrelationPairsMCRecSigSig", stringDDbar + "2D SigSig - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});" + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecSigBkg", stringDDbar + "2D SigBkg - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});" + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecBkgSig", stringDDbar + "2D BkgSig - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});" + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecBkgBkg", stringDDbar + "2D BkgBkg - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});" + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSignalRegionMCRec", stringMCReco + stringSignal + stringDeltaEta + "entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSignalRegionMCRec", stringMCReco + stringSignal + stringDeltaPhi + "entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSignalRegionMCRec", stringMCReco + stringSignal + stringDeltaPhi + stringDeltaEta + "entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaPtDDbarSignalRegionMCRec", stringMCReco + stringSignal + stringDeltaPt + "entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSignalRegionMCRec", stringMCReco + stringSignal + stringDeltaPtMaxMin + "entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hCorrel2DVsPtSignalRegionMCRecSigSig", stringMCReco + "SigSig" + stringSignal + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}},  //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecSigBkg", stringMCReco + "SigBkg" + stringSignal + +stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecBkgSig", stringMCReco + "BkgSig" + stringSignal + +stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecBkgBkg", stringMCReco + "BkgBkg" + stringSignal + +stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSidebandsMCRec", stringMCReco + stringSideband + stringDeltaEta + "entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSidebandsMCRec", stringMCReco + stringSideband + stringDeltaPhi + "entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSidebandsMCRec", stringMCReco + stringSideband + stringDeltaPhi + stringDeltaEta + "entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaPtDDbarSidebandsMCRec", stringMCReco + stringSideband + stringDeltaPt + "entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSidebandsMCRec", stringMCReco + stringSideband + stringDeltaPtMaxMin + "entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hCorrel2DVsPtSidebandsMCRecSigSig", stringMCReco + "SigSig" + stringSideband + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}},   //note: axes 3 and 4 (the pT) are updated in the init() - should be empty, kept for cross-check and debug
     {"hCorrel2DVsPtSidebandsMCRecSigBkg", stringMCReco + "SigBkg" + stringSideband + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}},   //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSidebandsMCRecBkgSig", stringMCReco + "BkgSig" + stringSideband + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}},   //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSidebandsMCRecBkgBkg", stringMCReco + "BkgBkg" + stringSideband + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}}}; //note: axes 3 and 4 (the pT) are updated in the init()

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{pTBinsCorrelations_v}, "pT bin limits for correlation plots"};
  //signal and sideband region edges, to be defined via json file (initialised to empty)
  Configurable<std::vector<double>> signalRegionInner{"signalRegionInner", std::vector<double>{signalRegionInner_v}, "Inner values of signal region vs pT"};
  Configurable<std::vector<double>> signalRegionOuter{"signalRegionOuter", std::vector<double>{signalRegionOuter_v}, "Outer values of signal region vs pT"};
  Configurable<std::vector<double>> sidebandLeftInner{"sidebandLeftInner", std::vector<double>{sidebandLeftInner_v}, "Inner values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandLeftOuter{"sidebandLeftOuter", std::vector<double>{sidebandLeftOuter_v}, "Outer values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightInner{"sidebandRightInner", std::vector<double>{sidebandRightInner_v}, "Inner values of right sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightOuter{"sidebandRightOuter", std::vector<double>{sidebandRightOuter_v}, "Outer values of right sideband vs pT"};

  void init(o2::framework::InitContext&)
  {
    // redefinition of pT axes for THnSparse holding correlation entries
    int nBinspTaxis = binsCorrelations->size() - 1;
    const double* valuespTaxis = binsCorrelations->data();

    for (int i = 2; i <= 3; i++) {
      registry.get<THnSparse>(HIST("hMass2DCorrelationPairsMCRecSigSig"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hMass2DCorrelationPairsMCRecSigBkg"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hMass2DCorrelationPairsMCRecBkgSig"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hMass2DCorrelationPairsMCRecBkgBkg"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSignalRegionMCRecSigSig"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSignalRegionMCRecSigBkg"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSignalRegionMCRecBkgSig"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSignalRegionMCRecBkgBkg"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSidebandsMCRecSigSig"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSidebandsMCRecSigBkg"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSidebandsMCRecBkgSig"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>(HIST("hCorrel2DVsPtSidebandsMCRecBkgBkg"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
    }
  }

  void process(aod::DDbarPairFull const& pairEntries)
  {
    for (auto& pairEntry : pairEntries) {
      //define variables for widely used quantities
      double deltaPhi = pairEntry.deltaPhi();
      double deltaEta = pairEntry.deltaEta();
      double ptD = pairEntry.ptD();
      double ptDbar = pairEntry.ptDbar();
      double massD = pairEntry.mD();
      double massDbar = pairEntry.mDbar();

      //reject entries outside pT ranges of interest
      int pTBinD = o2::analysis::findBin(binsCorrelations, ptD);
      int pTBinDbar = o2::analysis::findBin(binsCorrelations, ptDbar);
      if (pTBinD == -1 || pTBinDbar == -1) { //at least one particle outside accepted pT range
        continue;
      }

      //fill 2D invariant mass plots
      switch (pairEntry.signalStatus()) {
        case 0: //D Bkg, Dbar Bkg
          registry.fill(HIST("hMass2DCorrelationPairsMCRecBkgBkg"), massD, massDbar, ptD, ptDbar);
          break;
        case 1: //D Bkg, Dbar Sig
          registry.fill(HIST("hMass2DCorrelationPairsMCRecBkgSig"), massD, massDbar, ptD, ptDbar);
          break;
        case 2: //D Sig, Dbar Bkg
          registry.fill(HIST("hMass2DCorrelationPairsMCRecSigBkg"), massD, massDbar, ptD, ptDbar);
          break;
        case 3: //D Sig, Dbar Sig
          registry.fill(HIST("hMass2DCorrelationPairsMCRecSigSig"), massD, massDbar, ptD, ptDbar);
          break;
        default: //should not happen for MC reco
          break;
      }

      //check if correlation entry belongs to signal region, sidebands or is outside both, and fill correlation plots
      if (massD > signalRegionInner->at(pTBinD) && massD < signalRegionOuter->at(pTBinD) && massDbar > signalRegionInner->at(pTBinDbar) && massDbar < signalRegionOuter->at(pTBinDbar)) {
        //in signal region
        registry.fill(HIST("hCorrel2DPtIntSignalRegionMCRec"), deltaPhi, deltaEta);
        registry.fill(HIST("hDeltaEtaPtIntSignalRegionMCRec"), deltaEta);
        registry.fill(HIST("hDeltaPhiPtIntSignalRegionMCRec"), deltaPhi);
        registry.fill(HIST("hDeltaPtDDbarSignalRegionMCRec"), ptDbar - ptD);
        registry.fill(HIST("hDeltaPtMaxMinSignalRegionMCRec"), std::abs(ptDbar - ptD));
        switch (pairEntry.signalStatus()) {
          case 0: //D Bkg, Dbar Bkg
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecBkgBkg"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          case 1: //D Bkg, Dbar Sig
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecBkgSig"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          case 2: //D Sig, Dbar Bkg
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecSigBkg"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          case 3: //D Sig, Dbar Sig
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecSigSig"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          default: //should not happen for MC reco
            break;
        }
      }

      if ((massD > sidebandLeftInner->at(pTBinD) && massD < sidebandLeftOuter->at(pTBinD) && massDbar > sidebandLeftInner->at(pTBinDbar) && massDbar < sidebandRightOuter->at(pTBinDbar)) ||
          (massD > sidebandRightInner->at(pTBinD) && massD < sidebandRightOuter->at(pTBinD) && massDbar > sidebandLeftInner->at(pTBinDbar) && massDbar < sidebandRightOuter->at(pTBinDbar)) ||
          (massD > sidebandLeftInner->at(pTBinD) && massD < sidebandRightOuter->at(pTBinD) && massDbar > sidebandLeftInner->at(pTBinDbar) && massDbar < sidebandLeftOuter->at(pTBinDbar)) ||
          (massD > sidebandLeftInner->at(pTBinD) && massD < sidebandRightOuter->at(pTBinD) && massDbar > sidebandRightInner->at(pTBinDbar) && massDbar < sidebandRightOuter->at(pTBinDbar))) {
        //in sideband region
        registry.fill(HIST("hCorrel2DPtIntSidebandsMCRec"), deltaPhi, deltaEta);
        registry.fill(HIST("hDeltaEtaPtIntSidebandsMCRec"), deltaEta);
        registry.fill(HIST("hDeltaPhiPtIntSidebandsMCRec"), deltaPhi);
        registry.fill(HIST("hDeltaPtDDbarSidebandsMCRec"), ptDbar - ptD);
        registry.fill(HIST("hDeltaPtMaxMinSidebandsMCRec"), std::abs(ptDbar - ptD));
        switch (pairEntry.signalStatus()) {
          case 0: //D Bkg, Dbar Bkg
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecBkgBkg"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          case 1: //D Bkg, Dbar Sig
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecBkgSig"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          case 2: //D Sig, Dbar Bkg
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecSigBkg"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          case 3: //D Sig, Dbar Sig
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecSigSig"), deltaPhi, deltaEta, ptD, ptDbar);
            break;
          default: //should not happen for MC reco
            break;
        }
      }
    } //end loop
  }
};

/// D-Dbar correlation pair filling task, from pair tables - for MC gen-level analysis (no filter/selection, only true signal) - Ok for both USL and LS analyses
/// Works on both USL and LS analyses pair tables (and if tables are filled with quark pairs as well)
struct HfTaskCorrelationDDbarMcGen {

  HistogramRegistry registry{
    "registry",
    {{"hDeltaEtaPtIntMCGen", stringMCParticles + stringDeltaEta + "entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCGen", stringMCParticles + stringDeltaPhi + "entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCGen", stringMCParticles + stringDeltaPhi + stringDeltaEta + "entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtMCGen", stringMCParticles + stringDeltaPhi + stringDeltaEta + stringPtD + stringPtDbar + "entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarMCGen", stringMCParticles + stringDeltaPt + "entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCGen", stringMCParticles + stringDeltaPtMaxMin + "entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{pTBinsCorrelations_v}, "pT bin limits for correlation plots"};

  void init(o2::framework::InitContext&)
  {
    // redefinition of pT axes for THnSparse holding correlation entries
    int nBinspTaxis = binsCorrelations->size() - 1;
    const double* valuespTaxis = binsCorrelations->data();

    for (int i = 2; i <= 3; i++) {
      registry.get<THnSparse>(HIST("hCorrel2DVsPtMCGen"))->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
    }
  }

  void process(aod::DDbarPair const& pairEntries)
  {
    for (auto& pairEntry : pairEntries) {
      //define variables for widely used quantities
      double deltaPhi = pairEntry.deltaPhi();
      double deltaEta = pairEntry.deltaEta();
      double ptD = pairEntry.ptD();
      double ptDbar = pairEntry.ptDbar();

      //reject entries outside pT ranges of interest
      if (o2::analysis::findBin(binsCorrelations, ptD) == -1 || o2::analysis::findBin(binsCorrelations, ptDbar) == -1) {
        continue;
      }

      registry.fill(HIST("hCorrel2DVsPtMCGen"), deltaPhi, deltaEta, ptD, ptDbar);
      registry.fill(HIST("hCorrel2DPtIntMCGen"), deltaPhi, deltaEta);
      registry.fill(HIST("hDeltaEtaPtIntMCGen"), deltaEta);
      registry.fill(HIST("hDeltaPhiPtIntMCGen"), deltaPhi);
      registry.fill(HIST("hDeltaPtDDbarMCGen"), ptDbar - ptD);
      registry.fill(HIST("hDeltaPtMaxMinMCGen"), std::abs(ptDbar - ptD));
    } //end loop
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{};
  //MC-based tasks
  const bool doMCGen = cfgc.options().get<bool>("doMCGen");
  const bool doMCRec = cfgc.options().get<bool>("doMCRec");
  if (doMCGen) { //MC-Gen analysis
    workflow.push_back(adaptAnalysisTask<HfTaskCorrelationDDbarMcGen>(cfgc));
  } else if (doMCRec) { //MC-Rec analysis
    workflow.push_back(adaptAnalysisTask<HfTaskCorrelationDDbarMcRec>(cfgc));
  } else { //data analysis
    workflow.push_back(adaptAnalysisTask<HfTaskCorrelationDDbar>(cfgc));
  }
  return workflow;
}
