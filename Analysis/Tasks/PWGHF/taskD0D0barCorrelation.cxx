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
using namespace o2::aod::hf_d0d0bar_correlation;

namespace o2::aod
{
using D0D0barPairFull = soa::Join<aod::D0D0barPair, aod::D0D0barRecoInfo>;
} // namespace o2::aod

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  ConfigParamSpec optionDoMC{"doMC", VariantType::Bool, false, {"Run MC dedicated tasks."}};
  workflowOptions.push_back(optionDoMC);
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

/// D0-D0bar correlation pair filling task, from pair tables - for real data and data-like analysis (i.e. reco-level w/o matching request via MC truth)
/// Works on both USL and LS analyses pair tables
struct TaskD0D0barCorrelation {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 (from correlator task) for normalisation, and hMass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hMass2DCorrelationPairs", "D0,D0bar candidates 2D;inv. mass D0 (#pi K) (GeV/#it{c}^{2});inv. mass D0bar (#pi K) (GeV/#it{c}^{2});#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSignalRegion", "D0,D0bar candidates signal region;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSignalRegion", "D0,D0bar candidates signal region;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSignalRegion", "D0,D0bar candidates signal region;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSignalRegion", "D0,D0bar candidates signal region;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarSignalRegion", "D0,D0bar candidates signal region;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSignalRegion", "D0,D0bar candidates signal region;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hDeltaEtaPtIntSidebands", "D0,D0bar candidates sidebands;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSidebands", "D0,D0bar candidates sidebands;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSidebands", "D0,D0bar candidates sidebands;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSidebands", "D0,D0bar candidates sidebands;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaPtDDbarSidebands", "D0,D0bar candidates sidebands;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSidebands", "D0,D0bar candidates sidebands;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for correlation plots"};
  //signal and sideband region edges, to be defined via json file (initialised to empty)
  Configurable<std::vector<double>> signalRegionInner{"signalRegionInner", std::vector<double>(), "Inner values of signal region vs pT"};
  Configurable<std::vector<double>> signalRegionOuter{"signalRegionOuter", std::vector<double>(), "Outer values of signal region vs pT"};
  Configurable<std::vector<double>> sidebandLeftInner{"sidebandLeftInner", std::vector<double>(), "Inner values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandLeftOuter{"sidebandLeftOuter", std::vector<double>(), "Outer values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightInner{"sidebandRightInner", std::vector<double>(), "Inner values of right sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightOuter{"sidebandRightOuter", std::vector<double>(), "Outer values of right sideband vs pT"};

  // redefinition of pT axes for THnSparse holding correlation entries
  int nBinspTaxis = binsCorrelations->size() - 1;
  const double* valuespTaxis = binsCorrelations->data();

  void init(o2::framework::InitContext&) {
    for (int i=2; i<=3; i++) {
      registry.get<THnSparse>("hMass2DCorrelationPairs")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSignalRegion")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSidebands")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);     
    }
  }

  void process(aod::D0D0barPairFull const& pairEntries)
  {
    for (auto& pairEntry : pairEntries) {
      //fill 2D invariant mass plots
      registry.fill(HIST("hMass2DCorrelationPairs"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
      
      //check if correlation entry belongs to signal region, sidebands or is outside both, and fill correlation plots
      int pTBinD0 = o2::analysis::findBin(binsCorrelations, pairEntry.ptD0());
      int pTBinD0bar = o2::analysis::findBin(binsCorrelations, pairEntry.ptD0bar());

      if(pairEntry.mD0() > signalRegionInner->at(pTBinD0) && pairEntry.mD0() < signalRegionOuter->at(pTBinD0) && pairEntry.mD0bar() > signalRegionInner->at(pTBinD0bar) && pairEntry.mD0bar() < signalRegionOuter->at(pTBinD0bar)) {
        //in signal region
        registry.fill(HIST("hCorrel2DVsPtSignalRegion"), pairEntry.deltaPhi(), pairEntry.deltaEta(), pairEntry.ptD0(), pairEntry.ptD0bar());
        registry.fill(HIST("hCorrel2DPtIntSignalRegion"), pairEntry.deltaPhi(), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaEtaPtIntSignalRegion"), pairEntry.deltaPhi());
        registry.fill(HIST("hDeltaPhiPtIntSignalRegion"), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaPtDDbarSignalRegion"), pairEntry.ptD0bar() - pairEntry.ptD0());
        registry.fill(HIST("hDeltaPtMaxMinSignalRegion"), std::abs(pairEntry.ptD0bar() - pairEntry.ptD0()));
      }

      if((pairEntry.mD0() > sidebandLeftInner->at(pTBinD0) && pairEntry.mD0() < sidebandLeftOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandLeftInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandRightOuter->at(pTBinD0bar)) || 
         (pairEntry.mD0() > sidebandRightInner->at(pTBinD0) && pairEntry.mD0() < sidebandRightOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandLeftInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandRightOuter->at(pTBinD0bar)) ||
         (pairEntry.mD0() > sidebandLeftInner->at(pTBinD0) && pairEntry.mD0() < sidebandRightOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandLeftInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandLeftOuter->at(pTBinD0bar)) ||
         (pairEntry.mD0() > sidebandLeftInner->at(pTBinD0) && pairEntry.mD0() < sidebandRightOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandRightInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandRightOuter->at(pTBinD0bar))) {
        //in sideband region
        registry.fill(HIST("hCorrel2DVsPtSidebands"), pairEntry.deltaPhi(), pairEntry.deltaEta(), pairEntry.ptD0(), pairEntry.ptD0bar());
        registry.fill(HIST("hCorrel2DPtIntSidebands"), pairEntry.deltaPhi(), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaEtaPtIntSidebands"), pairEntry.deltaPhi());
        registry.fill(HIST("hDeltaPhiPtIntSidebands"), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaPtDDbarSidebands"), pairEntry.ptD0bar() - pairEntry.ptD0());
        registry.fill(HIST("hDeltaPtMaxMinSidebands"), std::abs(pairEntry.ptD0bar() - pairEntry.ptD0()));
      }
    } //end loop
  }
};

/// D0-D0bar correlation pair filling task, from pair tables - for MC reco-level analysis (candidates matched to true signal only, but also bkg sources are studied)
/// Works on both USL and LS analyses pair tables
struct TaskD0D0barCorrelationMCRec {

  HistogramRegistry registry{
    "registry",
    //NOTE: use hMassD0 (from correlator task) for normalisation, and hMass2DCorrelationPairs for 2D-sideband-subtraction purposes
    {{"hMass2DCorrelationPairsMCRecSigSig", "D0,D0bar candidates 2D SigSig - MC reco;inv. mass D0 (#pi K) (GeV/#it{c}^{2});inv. mass D0bar (#pi K) (GeV/#it{c}^{2});#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecSigBkg", "D0,D0bar candidates 2D SigBkg - MC reco;inv. mass D0 (#pi K) (GeV/#it{c}^{2});inv. mass D0bar (#pi K) (GeV/#it{c}^{2});#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecBkgSig", "D0,D0bar candidates 2D BkgSig - MC reco;inv. mass D0 (#pi K) (GeV/#it{c}^{2});inv. mass D0bar (#pi K) (GeV/#it{c}^{2});#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hMass2DCorrelationPairsMCRecBkgBkg", "D0,D0bar candidates 2D BkgBkg - MC reco;inv. mass D0 (#pi K) (GeV/#it{c}^{2});inv. mass D0bar (#pi K) (GeV/#it{c}^{2});#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{200, 1.6, 2.1}, {200, 1.6, 2.1}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hDeltaEtaPtIntSignalRegionMCRec", "D0,D0bar candidates signal region - MC reco;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSignalRegionMCRec", "D0,D0bar candidates signal region - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSignalRegionMCRec", "D0,D0bar candidates signal region - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hDeltaPtDDbarSignalRegionMCRec", "D0,D0bar candidates signal region - MC reco;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSignalRegionMCRec", "D0,D0bar candidates signal region - MC reco;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}},
     {"hCorrel2DVsPtSignalRegionMCRecSigSig", "D0,D0bar candidates signal region SigSig - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecSigBkg", "D0,D0bar candidates signal region SigBkg - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecBkgSig", "D0,D0bar candidates signal region BkgSig - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSignalRegionMCRecBkgBkg", "D0,D0bar candidates signal region BkgBkg - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()               
     {"hDeltaEtaPtIntSidebandsMCRec", "D0,D0bar candidates sidebands - MC reco;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntSidebandsMCRec", "D0,D0bar candidates sidebands - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntSidebandsMCRec", "D0,D0bar candidates sidebands - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtSidebandsMCRecSigSig", "D0,D0bar candidates sidebands SigSig - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init() - should be empty, kept for cross-check and debug
     {"hCorrel2DVsPtSidebandsMCRecSigBkg", "D0,D0bar candidates sidebands SigBkg - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSidebandsMCRecBkgSig", "D0,D0bar candidates sidebands BkgSig - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()
     {"hCorrel2DVsPtSidebandsMCRecBkgBkg", "D0,D0bar candidates sidebands BkgBkg - MC reco;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()               
     {"hDeltaPtDDbarSidebandsMCRec", "D0,D0bar candidates signal region - MC reco;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinSidebandsMCRec", "D0,D0bar candidates signal region - MC reco;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for correlation plots"};
  //signal and sideband region edges, to be defined via json file (initialised to empty)
  Configurable<std::vector<double>> signalRegionInner{"signalRegionInner", std::vector<double>(), "Inner values of signal region vs pT"};
  Configurable<std::vector<double>> signalRegionOuter{"signalRegionOuter", std::vector<double>(), "Outer values of signal region vs pT"};
  Configurable<std::vector<double>> sidebandLeftInner{"sidebandLeftInner", std::vector<double>(), "Inner values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandLeftOuter{"sidebandLeftOuter", std::vector<double>(), "Outer values of left sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightInner{"sidebandRightInner", std::vector<double>(), "Inner values of right sideband vs pT"};
  Configurable<std::vector<double>> sidebandRightOuter{"sidebandRightOuter", std::vector<double>(), "Outer values of right sideband vs pT"};

  // redefinition of pT axes for THnSparse holding correlation entries
  int nBinspTaxis = binsCorrelations->size() - 1;
  const double* valuespTaxis = binsCorrelations->data();

  void init(o2::framework::InitContext&) {
    for (int i=2; i<=3; i++) {
      registry.get<THnSparse>("hMass2DCorrelationPairsMCRecSigSig")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hMass2DCorrelationPairsMCRecSigBkg")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hMass2DCorrelationPairsMCRecBkgSig")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hMass2DCorrelationPairsMCRecBkgBkg")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSignalRegionMCRecSigSig")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSignalRegionMCRecSigBkg")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSignalRegionMCRecBkgSig")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSignalRegionMCRecBkgBkg")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSidebandsMCRecSigSig")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
      registry.get<THnSparse>("hCorrel2DVsPtSidebandsMCRecSigBkg")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);  
      registry.get<THnSparse>("hCorrel2DVsPtSidebandsMCRecBkgSig")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);       
      registry.get<THnSparse>("hCorrel2DVsPtSidebandsMCRecBkgBkg")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);  
    }
  }

  void process(aod::D0D0barPairFull const& pairEntries)
  {
    for (auto& pairEntry : pairEntries) {
      //fill 2D invariant mass plots
      switch (pairEntry.signalStatus()) {
        case 0: //D0 Bkg, D0bar Bkg
          registry.fill(HIST("hMass2DCorrelationPairsMCRecBkgBkg"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
          break;
        case 1: //D0 Bkg, D0bar Sig
          registry.fill(HIST("hMass2DCorrelationPairsMCRecBkgSig"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
          break;
        case 2: //D0 Sig, D0bar Bkg
          registry.fill(HIST("hMass2DCorrelationPairsMCRecSigBkg"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
          break;
        case 3: //D0 Sig, D0bar Sig
          registry.fill(HIST("hMass2DCorrelationPairsMCRecSigSig"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
          break;      
        default: //should not happen for MC reco
        break;
      }
     
      //check if correlation entry belongs to signal region, sidebands or is outside both, and fill correlation plots
      int pTBinD0 = o2::analysis::findBin(binsCorrelations, pairEntry.ptD0());
      int pTBinD0bar = o2::analysis::findBin(binsCorrelations, pairEntry.ptD0bar());

      if(pairEntry.mD0() > signalRegionInner->at(pTBinD0) && pairEntry.mD0() < signalRegionOuter->at(pTBinD0) && pairEntry.mD0bar() > signalRegionInner->at(pTBinD0bar) && pairEntry.mD0bar() < signalRegionOuter->at(pTBinD0bar)) {
        //in signal region
        registry.fill(HIST("hCorrel2DPtIntSignalRegionMCRec"), pairEntry.deltaPhi(), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaEtaPtIntSignalRegionMCRec"), pairEntry.deltaPhi());
        registry.fill(HIST("hDeltaPhiPtIntSignalRegionMCRec"), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaPtDDbarSignalRegionMCRec"), pairEntry.ptD0bar() - pairEntry.ptD0());
        registry.fill(HIST("hDeltaPtMaxMinSignalRegionMCRec"), std::abs(pairEntry.ptD0bar() - pairEntry.ptD0()));        
        switch (pairEntry.signalStatus()) {
          case 0: //D0 Bkg, D0bar Bkg
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecBkgBkg"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;
          case 1: //D0 Bkg, D0bar Sig
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecBkgSig"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;
          case 2: //D0 Sig, D0bar Bkg
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecSigBkg"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;
          case 3: //D0 Sig, D0bar Sig
            registry.fill(HIST("hCorrel2DVsPtSignalRegionMCRecSigSig"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;      
          default: //should not happen for MC reco
          break;
        }
      }

      if((pairEntry.mD0() > sidebandLeftInner->at(pTBinD0) && pairEntry.mD0() < sidebandLeftOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandLeftInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandRightOuter->at(pTBinD0bar)) || 
         (pairEntry.mD0() > sidebandRightInner->at(pTBinD0) && pairEntry.mD0() < sidebandRightOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandLeftInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandRightOuter->at(pTBinD0bar)) ||
         (pairEntry.mD0() > sidebandLeftInner->at(pTBinD0) && pairEntry.mD0() < sidebandRightOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandLeftInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandLeftOuter->at(pTBinD0bar)) ||
         (pairEntry.mD0() > sidebandLeftInner->at(pTBinD0) && pairEntry.mD0() < sidebandRightOuter->at(pTBinD0) && pairEntry.mD0bar() > sidebandRightInner->at(pTBinD0bar) && pairEntry.mD0bar() < sidebandRightOuter->at(pTBinD0bar))) {
        //in sideband region
        registry.fill(HIST("hCorrel2DPtIntSidebandsMCRec"), pairEntry.deltaPhi(), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaEtaPtIntSidebandsMCRec"), pairEntry.deltaPhi());
        registry.fill(HIST("hDeltaPhiPtIntSidebandsMCRec"), pairEntry.deltaEta());
        registry.fill(HIST("hDeltaPtDDbarSidebandsMCRec"), pairEntry.ptD0bar() - pairEntry.ptD0());
        registry.fill(HIST("hDeltaPtMaxMinSidebandsMCRec"), std::abs(pairEntry.ptD0bar() - pairEntry.ptD0()));        
        switch (pairEntry.signalStatus()) {
          case 0: //D0 Bkg, D0bar Bkg
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecBkgBkg"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;
          case 1: //D0 Bkg, D0bar Sig
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecBkgSig"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;
          case 2: //D0 Sig, D0bar Bkg
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecSigBkg"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;
          case 3: //D0 Sig, D0bar Sig
            registry.fill(HIST("hCorrel2DVsPtSidebandsMCRecSigSig"), pairEntry.mD0(), pairEntry.mD0bar(), pairEntry.ptD0(), pairEntry.ptD0bar());
            break;      
          default: //should not happen for MC reco
          break;
        }
      }
    } //end loop
  }
};

/// D0-D0bar correlation pair filling task, from pair tables - for MC gen-level analysis (no filter/selection, only true signal) - Ok for both USL and LS analyses
/// Works on both USL and LS analyses pair tables (and if tables are filled with quark pairs as well)
struct TaskD0D0barCorrelationMCGen {

  HistogramRegistry registry{
    "registry",
    {{"hDeltaEtaPtIntMCGen", "D0,D0bar particles - MC gen;#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH1F, {{200, -10., 10.}}}},
     {"hDeltaPhiPtIntMCGen", "D0,D0bar particles - MC gen;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hCorrel2DPtIntMCGen", "D0,D0bar particles - MC gen;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};entries", {HistType::kTH2F, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {200, -10., 10.}}}},
     {"hCorrel2DVsPtMCGen", "D0,D0bar particles - MC gen;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};#it{#eta}^{D0bar}-#it{#eta}^{D0};#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};entries", {HistType::kTHnSparseD, {{32, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}, {120, -6., 6.}, {10, 0., 10.}, {10, 0., 10.}}}}, //note: axes 3 and 4 (the pT) are updated in the init()     
     {"hDeltaPtDDbarMCGen", "D0,D0bar particles - MC gen;#it{p}_{T}^{D0bar}-#it{p}_{T}^{D0};entries", {HistType::kTH1F, {{144, -36., 36.}}}},
     {"hDeltaPtMaxMinMCGen", "D0,D0bar particles - MC gen;#it{p}_{T}^{max}-#it{p}_{T}^{min};entries", {HistType::kTH1F, {{72, 0., 36.}}}}}};

  //pT ranges for correlation plots: the default values are those embedded in hf_cuts_d0_topik (i.e. the mass pT bins), but can be redefined via json files
  Configurable<std::vector<double>> binsCorrelations{"ptBinsForCorrelations", std::vector<double>{o2::analysis::hf_cuts_d0_topik::pTBins_v}, "pT bin limits for correlation plots"};

  // redefinition of pT axes for THnSparse holding correlation entries
  int nBinspTaxis = binsCorrelations->size() - 1;
  const double* valuespTaxis = binsCorrelations->data();

  void init(o2::framework::InitContext&) {
    for (int i=2; i<=3; i++) {
      registry.get<THnSparse>("hCorrel2DVsPtMCGen")->GetAxis(i)->Set(nBinspTaxis, valuespTaxis);
    }
  }

  void process(aod::D0D0barPairFull const& pairEntries)
  {
    for (auto& pairEntry : pairEntries) {
      registry.fill(HIST("hCorrel2DVsPtMCGen"), pairEntry.deltaPhi(), pairEntry.deltaEta(), pairEntry.ptD0(), pairEntry.ptD0bar());
      registry.fill(HIST("hCorrel2DPtIntMCGen"), pairEntry.deltaPhi(), pairEntry.deltaEta());
      registry.fill(HIST("hDeltaEtaPtIntMCGen"), pairEntry.deltaPhi());
      registry.fill(HIST("hDeltaPhiPtIntMCGen"), pairEntry.deltaEta());
      registry.fill(HIST("hDeltaPtDDbarMCGen"), pairEntry.ptD0bar() - pairEntry.ptD0());
      registry.fill(HIST("hDeltaPtMaxMinMCGen"), std::abs(pairEntry.ptD0bar() - pairEntry.ptD0()));
    } //end loop
  }
};

/// checks phi resolution for standard definition and sec-vtx based definition
struct TaskD0D0barCorrelationCheckPhiResolution {

  HistogramRegistry registry{
    "registry",
    {{"hMass", "D0,D0bar candidates;inv. mass (#pi K) (GeV/#it{c}^{2});entries", {HistType::kTH1F, {{120, 1.5848, 2.1848}}}},
     {"hEta", "D0,D0bar candidates;candidate #it{#eta};entries", {HistType::kTH1F, {{100, -5., 5.}}}},
     {"hPhiStdPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, 0., 2. * o2::constants::math::PI}, {50, 0., 50.}}}},
     {"hPhiByVtxPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, 0., 2. * o2::constants::math::PI}, {50, 0., 50.}}}},
     {"hPhiDifferenceTwoMethods", "D0,D0bar candidates;candidate #it{#Delta#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, -o2::constants::math::PI, o2::constants::math::PI}, {50, 0., 50.}}}},
     {"hDifferenceGenPhiStdPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, -o2::constants::math::PI, o2::constants::math::PI}, {50, 0., 50.}}}},
     {"hDifferenceGenPhiByVtxPhi", "D0,D0bar candidates;candidate #it{#varphi};#it{p}_{T}", {HistType::kTH2F, {{128, -o2::constants::math::PI, o2::constants::math::PI}, {50, 0., 50.}}}},
     {"hDeltaPhiPtIntStdPhi", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPhiPtIntByVtxPhi", "D0,D0bar candidates;#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH1F, {{128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPhiVsPtStdPhi", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{36, 0., 36.}, {36, 0., 36.}, {128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}},
     {"hDeltaPhiVsPtByVtxPhi", "D0,D0bar candidates;#it{p}_{T}^{D0};#it{p}_{T}^{D0bar};#it{#varphi}^{D0bar}-#it{#varphi}^{D0};entries", {HistType::kTH3F, {{36, 0., 36.}, {36, 0., 36.}, {128, -o2::constants::math::PI / 2., 3. * o2::constants::math::PI / 2.}}}}}};

  Configurable<int> dSelectionFlagD0{"dSelectionFlagD0", 1, "Selection Flag for D0"};
  Configurable<int> dSelectionFlagD0bar{"dSelectionFlagD0bar", 1, "Selection Flag for D0bar"};
  Configurable<double> cutYCandMax{"cutYCandMax", -1., "max. cand. rapidity"};
  Configurable<double> cutPtCandMin{"cutPtCandMin", -1., "min. cand. pT"};

  Filter filterSelectCandidates = (aod::hf_selcandidate_d0::isSelD0 >= dSelectionFlagD0 || aod::hf_selcandidate_d0::isSelD0bar >= dSelectionFlagD0bar);

  void process(aod::Collision const& collision, soa::Filtered<soa::Join<aod::HfCandProng2, aod::HFSelD0Candidate, aod::HfCandProng2MCRec>> const& candidates, aod::McParticles const& particlesMC, aod::BigTracksMC const& tracksMC)
  {
    for (auto& candidate1 : candidates) {
      //check decay channel flag for candidate1
      if (!(candidate1.hfflag() & 1 << D0ToPiK)) {
        continue;
      }
      if (cutYCandMax >= 0. && std::abs(YD0(candidate1)) > cutYCandMax) {
        continue;
      }
      if (cutPtCandMin >= 0. && candidate1.pt() < cutPtCandMin) {
        continue;
      }
      registry.fill(HIST("hMass"), InvMassD0(candidate1));
      registry.fill(HIST("hEta"), candidate1.eta());

      //D-Dbar correlation dedicated section
      //if it's a candidate D0, search for D0bar and evaluate correlations
      if (candidate1.isSelD0() >= dSelectionFlagD0) {
        double xPrimaryVertex = candidate1.index0_as<aod::BigTracksMC>().collision().posX(), yPrimaryVertex = candidate1.index0_as<aod::BigTracksMC>().collision().posY();
        double pt1 = candidate1.pt(), phi1Std = candidate1.phi();
        double phi1ByVtx = evaluatePhiByVertex(xPrimaryVertex, candidate1.xSecondaryVertex(), yPrimaryVertex, candidate1.ySecondaryVertex());
        registry.fill(HIST("hPhiStdPhi"), phi1Std, pt1);
        registry.fill(HIST("hPhiByVtxPhi"), phi1ByVtx, pt1);
        registry.fill(HIST("hPhiDifferenceTwoMethods"), getDeltaPhiForResolution(phi1ByVtx, phi1Std), pt1);

        //get corresponding gen-level D0, if exists, and evaluate gen-rec phi-difference with two approaches
        if (std::abs(candidate1.flagMCMatchRec()) == 1 << D0ToPiK) {                                                          //ok to keep both D0 and D0bar
          int indexGen = RecoDecay::getMother(particlesMC, candidate1.index0_as<aod::BigTracksMC>().mcParticle(), 421, true); //MC-gen corresponding index for MC-reco candidate
          double phi1Gen = particlesMC.iteratorAt(indexGen).phi();
          registry.fill(HIST("hDifferenceGenPhiStdPhi"), getDeltaPhiForResolution(phi1Std, phi1Gen), pt1);
          registry.fill(HIST("hDifferenceGenPhiByVtxPhi"), getDeltaPhiForResolution(phi1ByVtx, phi1Gen), pt1);
        }

        for (auto& candidate2 : candidates) {
          //check decay channel flag for candidate2
          if (!(candidate2.hfflag() & 1 << D0ToPiK)) {
            continue;
          }
          if (candidate2.isSelD0bar() >= dSelectionFlagD0bar) { //accept only D0bar candidates
            if (cutYCandMax >= 0. && std::abs(YD0(candidate2)) > cutYCandMax) {
              continue;
            }
            if (cutPtCandMin >= 0. && candidate2.pt() < cutPtCandMin) {
              continue;
            }
            //Excluding self-correlations (could happen in case of reflections)
            if (candidate1.mRowIndex == candidate2.mRowIndex) {
              continue;
            }
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
  WorkflowSpec workflow{};
  //MC-based tasks
  const bool doMCGen = cfgc.options().get<bool>("doMCGen");
  const bool doMCRec = cfgc.options().get<bool>("doMCRec");
  if (doMCGen) { //MC-Gen analysis
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelation>(cfgc));
  } else if (doMCRec) { //MC-Rec analysis
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationMCRec>(cfgc));
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationCheckPhiResolution>(cfgc));
  } else { //data analysis
    workflow.push_back(adaptAnalysisTask<TaskD0D0barCorrelationMCGen>(cfgc));
  }
  return workflow;
}
