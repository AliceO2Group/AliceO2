// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/*
I ran this code using following:
o2-analysis-timestamp| o2-analysis-upc-forward | o2-analysis-event-selection  --aod-file <path to ao2d.txt> [--isPbPb] -b
for now AO2D.root I am using is
alien:///alice/data/2015/LHC15o/000246392/pass5_lowIR/PWGZZ/Run3_Conversion/148_20210304-0829_child_1/AOD/001/AO2D.root
*/
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/EventSelection.h"
#include <TH1D.h>
#include <TH2D.h>
#include <TString.h>
#include "TLorentzVector.h"
#include "AnalysisCore/TriggerAliases.h"
using namespace std;
using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
#define mmuon 0.1057 //mass of muon
struct UPCForward {
  //defining histograms using histogram registry
  HistogramRegistry registry{
    "registry",
    {{"hMass", ";#it{m_{#mu#mu}}, GeV/c^{2};", {HistType::kTH1D, {{500, 0, 10}}}},
     {"hPt", ";#it{p_{t}}, GeV/c;", {HistType::kTH1D, {{500, 0., 5.}}}},
     {"hPtsingle_muons", ";#it{p_{t}}, GeV/c;", {HistType::kTH1D, {{500, 0., 5.}}}},
     {"hPx", ";#it{p_{x}}, GeV/c;", {HistType::kTH1D, {{500, -5., 5.}}}},
     {"hPy", ";#it{p_{y}}, GeV/c;", {HistType::kTH1D, {{500, -5., 5.}}}},
     {"hPz", ";#it{p_{z}}, GeV/c;", {HistType::kTH1D, {{500, -5., 5.}}}},
     {"hRap", ";#it{y},..;", {HistType::kTH1D, {{500, -10., 10.}}}},
     {"hEta", ";#it{y},..;", {HistType::kTH1D, {{500, -10., 10.}}}},
     {"hCharge", ";#it{charge},..;", {HistType::kTH1D, {{500, -10., 10.}}}},
     {"hSelectionCounter", ";#it{Selection},..;", {HistType::kTH1I, {{30, 0., 30.}}}},
     {"hPhi", ";#it{#Phi},;", {HistType::kTH1D, {{500, -6., 6.}}}}}};

  void init(o2::framework::InitContext&)
  {
    //TString SelectionCuts[8] = {"NoSelection", "CMup11and10Trigger","V0Selection","FDSelection", "twotracks", "oppositecharge", "-2.5<Eta<-4", "Pt<1"};

    /* // commenting out this part now as histogram registry do not seem to suppor the binlabel using AxisSpec() or any other method
    for (int i = 0; i < 6; i++) {
      registry.GetXaxis().(HIST("hSelectionCounter",SetBinLabel(i + 1, SelectionCuts[i].Data())));
    }*/
  }
  //new
  void process(soa::Join<aod::BCs, aod::BcSels>::iterator const& bc, aod::Muons const& tracksMuon)
  {
    registry.fill(HIST("hSelectionCounter"), 0);

    int iMuonTracknumber = 0;
    TLorentzVector p1, p2, p;
    bool ispositive = kFALSE;
    bool isnegative = kFALSE;

    // V0 and FD information
    bool isBeamBeamV0A = bc.bbV0A();
    bool isBeamGasV0A = bc.bgV0A();
    bool isBeamBeamV0C = bc.bbV0C();
    bool isBeamGasV0C = bc.bgV0C();

    bool isBeamBeamFDA = bc.bbFDA();
    bool isBeamGasFDA = bc.bgFDA();
    bool isBeamBeamFDC = bc.bbFDC();
    bool isBeamGasFDC = bc.bgFDC();

    //offline V0 and FD selection
    bool isV0Selection = isBeamBeamV0A || isBeamGasV0A || isBeamGasV0C;
    bool isFDSelection = isBeamBeamFDA || isBeamGasFDA || isBeamBeamFDC || isBeamGasFDC;

    //CCUP10 and CCUP11 information
    bool iskMUP11fired = bc.alias()[kMUP11];
    bool iskMUP10fired = bc.alias()[kMUP10];

    // selecting kMUP10 and 11 triggers
    if (!iskMUP11fired && !iskMUP10fired) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 1);

    if (isV0Selection) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 2);

    if (isFDSelection) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 3);

    for (auto& muon : tracksMuon) {
      registry.fill(HIST("hCharge"), muon.sign());
      iMuonTracknumber++;
      if (muon.sign() > 0) {
        p1.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
        ispositive = kTRUE;
      }
      if (muon.sign() < 0) {
        p2.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
        isnegative = kTRUE;
      }
    }
    if (iMuonTracknumber != 2) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 4);
    if (!ispositive || !isnegative) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 5);

    if (-4 < p1.Eta() < -2.5 || -4 < p2.Eta() < -2.5) {

      return;
    }
    registry.fill(HIST("hSelectionCounter"), 6);
    p = p1 + p2;
    if (p.Pt() > 1) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 7);
    registry.fill(HIST("hPt"), p.Pt());
    registry.fill(HIST("hPx"), p.Px());
    registry.fill(HIST("hPy"), p.Py());
    registry.fill(HIST("hPz"), p.Pz());
    registry.fill(HIST("hRap"), p.Rapidity());
    registry.fill(HIST("hMass"), p.M());
    registry.fill(HIST("hPhi"), p.Phi());
    registry.fill(HIST("hEta"), p1.Eta());
    registry.fill(HIST("hEta"), p2.Eta());
    registry.fill(HIST("hPtsingle_muons"), p1.Pt());
    registry.fill(HIST("hPtsingle_muons"), p2.Pt());

  } //end of process
};
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCForward>(cfgc)};
}
