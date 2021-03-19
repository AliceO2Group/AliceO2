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
o2-analysis-timestamp| o2-analysis-upc-forward  --aod-file <path to ao2d.txt> [--isPbPb]  --aod-memory-rate-limit 10000000000 --shm-segment-size 16000000000 -b
for now AO2D.root I am using is
/alice/data/2015/LHC15o/000246392/pass5_lowIR/PWGZZ/Run3_Conversion/138_20210129-0800_child_1/0001/AO2D.root
it can be copied using
alien_cp alien:/alice/data/2015/LHC15o/000246392/pass5_lowIR/PWGZZ/Run3_Conversion/138_20210129-0800_child_1/0001/AO2D.root  file:./
*/
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
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
  //defining histograms
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
    TString SelectionCuts[6] = {"NoSelection", "CMup11Trigger", "twotracks", "oppositecharge", "-2.5<Eta<-4", "Pt<1"};

    /* // commenting out this part now as histogram registry do not seem to suppor the binlabel usign AxisSpec() or any other method
    for (int i = 0; i < 6; i++) {
      registry.GetXaxis().(HIST("hSelectionCounter",SetBinLabel(i + 1, SelectionCuts[i].Data())));
    }*/
  }
  void process(soa::Join<aod::BCs, aod::Run2BCInfos>::iterator const& bc, aod::Muons const& tracksMuon)
  {
    registry.fill(HIST("hSelectionCounter"), 0);

    int iMuonTracknumber = 0;
    TLorentzVector p1, p2, p;
    bool ispositive = kFALSE;
    bool isnegative = kFALSE;
    /*this code below is suggested by evgeny.
    this code is now hardcoded for runs  246391, 246392 for CMUP11
    and 244980, 244982, 244983, 245064, 245066, 245068 for CMUP10*/
    uint64_t classIndexMUP = -1;
    Int_t iRunNumber = bc.runNumber();

    if (iRunNumber == 246391 || iRunNumber == 246392) {
      classIndexMUP = 51; //CMUP11
    } else if (iRunNumber == 246980 || iRunNumber == 246982 || iRunNumber == 246983) {
      classIndexMUP = 88; //CMUP10
    } else if (iRunNumber == 245064 || iRunNumber == 245066 || iRunNumber == 245068) {
      classIndexMUP = 62; //CMUP10
    }
    if (classIndexMUP == -1) {
      return;
    }
    //selecting CMUP10 and CMUP11 events selection
    bool isMUP11fired = bc.triggerMaskNext50() & (1ull << classIndexMUP - 50);

    if (!isMUP11fired) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 1);
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
    registry.fill(HIST("hSelectionCounter"), 2);
    if (!ispositive || !isnegative) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 3);

    if (-4 < p1.Eta() < -2.5 || -4 < p2.Eta() < -2.5) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 4);
    p = p1 + p2;
    if (p.Pt() > 1) {
      return;
    }
    registry.fill(HIST("hSelectionCounter"), 5);
    registry.fill(HIST("hPt"), p.Pt());
    registry.fill(HIST("hPx"), p.Px());
    registry.fill(HIST("hPy"), p.Py());
    registry.fill(HIST("hPz"), p.Pz());
    registry.fill(HIST("hRap"), p.Rapidity());
    registry.fill(HIST("hMass"), p.M());
    registry.fill(HIST("hPhi"), p.Phi());
    registry.fill(HIST("hEta"), p1.Eta());
    registry.fill(HIST("hEta"), p2.Eta());
    registry.fill(HIST("hPtsingle"), p1.Pt());
    registry.fill(HIST("hPtsingle"), p2.Pt());

  } //end of process
};
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCForward>(cfgc, TaskName{"upc-forward"})};
}
