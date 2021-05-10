// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file taskDDbarCorrelation.cxx
/// \brief D-Dbar analysis task - data-like, MC-reco and MC-kine analyses. For ULS and LS pairs
///
/// \author Fabio Colamaria <fabio.colamaria@ba.infn.it>, INFN Bari

#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisCore/HFSelectorCuts.h"
#include "AnalysisDataModel/HFSecondaryVertex.h"
#include "AnalysisDataModel/HFCandidateSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_prong2;
using namespace o2::analysis::hf_cuts_d0_topik;
using namespace o2::framework::expressions;
using namespace o2::constants::math;
using namespace o2::aod::hf_ddbar_correlation;

namespace o2::aod
{
using DDbarPairFull = soa::Join<aod::DDbarPair, aod::DDbarRecoInfo>;
} // namespace o2::aod

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMCGen{"doMCGen", VariantType::Bool, false, {"Run MC-Gen dedicated tasks."}};
  ConfigParamSpec optionDoMCRec{"doMCRec", VariantType::Bool, false, {"Run MC-Rec dedicated tasks."}};
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
/// Returns deltaPhi value in range [-pi, pi], for resolution distributions
///
double getDeltaPhiForResolution(double phiD, double phiDbar)
{
  return RecoDecay::constrainAngle(phiDbar - phiD, -o2::constants::math::PI);
}

///
/// Returns phi of candidate/particle evaluated from x and y components of segment connecting primary and secondary vertices
///
double evaluatePhiByVertex(double xVertex1, double xVertex2, double yVertex1, double yVertex2)
{
  return RecoDecay::Phi(xVertex2 - xVertex1, yVertex2 - yVertex1);
}
/*
/// definition of axes for THnSparse (dPhi,dEta,pTD,pTDbar) - note: last two axis are dummy, will be replaced later
const int nBinsSparse[4] = {32,120,10,10}; 
const double binMinSparse[4] = {-o2::constants::math::PI / 2.,-6.,0.,0.;  //is the minimum for all the bins
const double binMaxSparse[4] = {3. * o2::constants::math::PI / 2.,6.,10.,10.};  //is the maximum for all the bins
*/

/// D-Dbar correlation pair filling task, from pair tables - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
/// Works on both USL and LS analyses pair tables
struct TaskDDbarCorrelation {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 (from correlator task) for normalisation, and hMass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hMass2DCorrelationPairs", "D,Dbar candidates 2D;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSignalRegion", "D,Dbar candidates signal region;#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSignalRegion", "D,Dbar candidates signal region;#it{#varphi}^{Dbar}-#it{#varphi}^{D};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSignalRegion", "D,Dbar candidates signal region;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSignalRegion", "D,Dbar candidates signal region;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarSignalRegion", "D,Dbar candidates signal region;#it{p}_{T}^{Dbar}-#it{p}_{T}^{D};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSignalRegion", "D,Dbar candidates signal region;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hDeltaEtaPtIntSidebands", "D,Dbar candidates sidebands;#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSidebands", "D,Dbar candidates sidebands;#it{#varphi}^{Dbar}-#it{#varphi}^{D};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSidebands", "D,Dbar candidates sidebands;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSidebands", "D,Dbar candidates sidebands;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarSidebands", "D,Dbar candidates sidebands;#it{p}_{T}^{Dbar}-#it{p}_{T}^{D};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSidebands", "D,Dbar candidates sidebands;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for correlation plots"};
  //signal and sideband region edges, to be defined via json file (initialised to empty)
  Configurable<std::vector<double>> signalRegionInner{"signalRegionInner", std::vector<double>(), "Inner values of signal region vs pT"};
  Configurable<std::vector<double>> signalRegionOuter{"signalRegionOuter", std::vector<double>(), "Outer values of signal region vs pT"};
  Configurable<std::vector<double>> sidebandLeftInner{"sidebandLeftInner", std::vector<double>(), "Inner values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandLeftOuter{"sidebandLeftOuter", std::vector<double>(), "Outer values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightInner{"sidebandRightInner", std::vector<double>(), "Inner values of right sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightOuter{"sidebandRightOuter", std::vector<double>(), "Outer values of right sideband vs pT"};

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

      //reject entries outside pT ranges of interest
      double minPtAllowed = binsCorrelations->at(0);
      double maxPtAllowed = binsCorrelations->at(binsCorrelations->size() - 1);
      if (ptD < minPtAllowed || ptDbar < minPtAllowed || ptD > maxPtAllowed || ptDbar > maxPtAllowed) {
        continue;
      }

      //fill 2D invariant mass plots
      registry.fill(HIST("hMass2DCorrelationPairs"), massD, massDbar, ptD, ptDbar);

      //check if correlation entry belongs to signal region, sidebands or is outside both, and fill correlation plots
      int pTBinD = o2::analysis::findBin(binsCorrelations, ptD);
      int pTBinDbar = o2::analysis::findBin(binsCorrelations, ptDbar);

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
struct TaskDDbarCorrelationMCRec {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 (from correlator task) for normalisation, and hMass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hMass2DCorrelationPairsMCRecSigSig", "D,Dbar candidates 2D SigSig - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecSigBkg", "D,Dbar candidates 2D SigBkg - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecBkgSig", "D,Dbar candidates 2D BkgSig - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecBkgBkg", "D,Dbar candidates 2D BkgBkg - MC reco;inv. mass D (GeV/#it{c}^{2});inv. mass Dbar (GeV/#it{c}^{2});#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSignalRegionMCRec", "D,Dbar candidates signal region - MC reco;#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSignalRegionMCRec", "D,Dbar candidates signal region - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSignalRegionMCRec", "D,Dbar candidates signal region - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaPtDDbarSignalRegionMCRec", "D,Dbar candidates signal region - MC reco;#it{p}_{T}^{Dbar}-#it{p}_{T}^{D};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSignalRegionMCRec", "D,Dbar candidates signal region - MC reco;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hCorrel2DVsPtSignalRegionMCRecSigSig", "D,Dbar candidates signal region SigSig - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecSigBkg", "D,Dbar candidates signal region SigBkg - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecBkgSig", "D,Dbar candidates signal region BkgSig - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecBkgBkg", "D,Dbar candidates signal region BkgBkg - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSidebandsMCRec", "D,Dbar candidates sidebands - MC reco;#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSidebandsMCRec", "D,Dbar candidates sidebands - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSidebandsMCRec", "D,Dbar candidates sidebands - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSidebandsMCRecSigSig", "D,Dbar candidates sidebands SigSig - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init() - should be empty, kept for cross-check and debug
     {"hCorrel2DVsPtSidebandsMCRecSigBkg", "D,Dbar candidates sidebands SigBkg - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSidebandsMCRecBkgSig", "D,Dbar candidates sidebands BkgSig - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSidebandsMCRecBkgBkg", "D,Dbar candidates sidebands BkgBkg - MC reco;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarSidebandsMCRec", "D,Dbar candidates signal region - MC reco;#it{p}_{T}^{Dbar}-#it{p}_{T}^{D};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSidebandsMCRec", "D,Dbar candidates signal region - MC reco;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for correlation plots"};
  //signal and sideband region edges, to be defined via json file (initialised to empty)
  Configurable<std::vector<double>> signalRegionInner{"signalRegionInner", std::vector<double>(), "Inner values of signal region vs pT"};
  Configurable<std::vector<double>> signalRegionOuter{"signalRegionOuter", std::vector<double>(), "Outer values of signal region vs pT"};
  Configurable<std::vector<double>> sidebandLeftInner{"sidebandLeftInner", std::vector<double>(), "Inner values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandLeftOuter{"sidebandLeftOuter", std::vector<double>(), "Outer values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightInner{"sidebandRightInner", std::vector<double>(), "Inner values of right sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightOuter{"sidebandRightOuter", std::vector<double>(), "Outer values of right sideband vs pT"};

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
      double minPtAllowed = binsCorrelations->at(0);
      double maxPtAllowed = binsCorrelations->at(binsCorrelations->size() - 1);
      if (ptD < minPtAllowed || ptDbar < minPtAllowed || ptD > maxPtAllowed || ptDbar > maxPtAllowed) {
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
      int pTBinD = o2::analysis::findBin(binsCorrelations, ptD);
      int pTBinDbar = o2::analysis::findBin(binsCorrelations, ptDbar);

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
struct TaskDDbarCorrelationMCGen {

  HistogramRegistry registry{
    "registry",
    {{"hDeltaEtaPtIntMCGen", "D,Dbar particles - MC gen;#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCGen", "D,Dbar particles - MC gen;#it{#varphi}^{Dbar}-#it{#varphi}^{D};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCGen", "D,Dbar particles - MC gen;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtMCGen", "D,Dbar particles - MC gen;#it{#varphi}^{Dbar}-#it{#varphi}^{D};#it{#eta}^{Dbar}-#it{#eta}^{D};#it{p}_{T}^{D};#it{p}_{T}^{Dbar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarMCGen", "D,Dbar particles - MC gen;#it{p}_{T}^{Dbar}-#it{p}_{T}^{D};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCGen", "D,Dbar particles - MC gen;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for correlation plots"};

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
      double minPtAllowed = binsCorrelations->at(0);
      double maxPtAllowed = binsCorrelations->at(binsCorrelations->size() - 1);
      if (ptD < minPtAllowed || ptDbar < minPtAllowed || ptD > maxPtAllowed || ptDbar > maxPtAllowed) {
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
    workflow.push_back(adaptAnalysisTask<TaskDDbarCorrelationMCGen>(cfgc, TaskName{"task-ddbar-correlation-mc-gen"}));
  } else if (doMCRec) { //MC-Rec analysis
    workflow.push_back(adaptAnalysisTask<TaskDDbarCorrelationMCRec>(cfgc, TaskName{"task-ddbar-correlation-mc-rec"}));
  } else { //data analysis
    workflow.push_back(adaptAnalysisTask<TaskDDbarCorrelation>(cfgc, TaskName{"task-ddbar-correlation"}));
  }
  return workflow;
}
