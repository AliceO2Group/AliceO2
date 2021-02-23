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
  OutputObj<TH1D> hMass{TH1D("hMass", ";#it{m_{#mu#mu}}, GeV/c^{2};", 500, 0, 10)};
  OutputObj<TH1D> hPt{TH1D("hPt", ";#it{p_{t}}, GeV/c;", 500, 0., 5.)};
  OutputObj<TH1D> hPx{TH1D("hPx", ";#it{p_{x}}, GeV/c;", 500, 0., 5.)};
  OutputObj<TH1D> hPy{TH1D("hPy", ";#it{p_{y}}, GeV/c;", 500, 0., 5.)};
  OutputObj<TH1D> hPz{TH1D("hPz", ";#it{p_{z}}, GeV/c;", 500, 0., 5.)};
  OutputObj<TH1D> hRap{TH1D("hRap", ";#it{y},..;", 500, -10., 10.)};
  OutputObj<TH1D> hCharge{TH1D("hCharge", ";#it{charge},..;", 500, -10., 10.)};
  OutputObj<TH1D> hSelectionCounter{TH1D("hSelectionCounter", ";#it{Selection},..;", 30, 0., 30.)};
  OutputObj<TH1D> hPhi{TH1D("hPhi", ";#it{#Phi},;", 500, -6., 6.)};

  void init(o2::framework::InitContext&)
  {
    TString SelectionCuts[6] = {"NoSelection", "CMup11Trigger", "twotracks", "oppositecharge", "-2.5<Eta<-4", "Pt<1"};
    for (int i = 0; i < 6; i++) {
      hSelectionCounter->GetXaxis()->SetBinLabel(i + 1, SelectionCuts[i].Data());
    }
  }
  void process(aod::BC const& bc, aod::Muons const& tracksMuon)
  {
    hSelectionCounter->Fill(0);

    int iMuonTracknumber = 0;
    TLorentzVector p1, p2, p;
    bool ispositive = kFALSE;
    bool isnegative = kFALSE;
    /*this code below is suggested by evgeny.
    this code is now hardcoded for run 246392 as we are not sure if trigger id is same for all the runs*/
    uint64_t classIndexMUP11 = 54; // 246392
    bool isMUP11fired = bc.triggerMask() & (1ull << classIndexMUP11);
    if (!isMUP11fired) {
      return;
    }
    LOGF(info, "mup11 fired");
    hSelectionCounter->Fill(1);
    for (auto& muon : tracksMuon) {
      hCharge->Fill(muon.charge());
      iMuonTracknumber++;
      if (muon.charge() > 0) {
        p1.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
        ispositive = kTRUE;
      }
      if (muon.charge() < 0) {
        p2.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
        isnegative = kTRUE;
      }
    }
    if (iMuonTracknumber != 2) {
      return;
    }
    hSelectionCounter->Fill(2);
    if (!ispositive || !isnegative) {
      return;
    }
    hSelectionCounter->Fill(3);

    if (-4 < p1.Eta() < -2.5 || -4 < p2.Eta() < -2.5) {
      return;
    }
    hSelectionCounter->Fill(4);
    p = p1 + p2;
    if (p.Pt() > 1) {
      return;
    }
    hSelectionCounter->Fill(5);
    hPt->Fill(p.Pt());
    hPx->Fill(p.Px());
    hPy->Fill(p.Py());
    hPz->Fill(p.Pz());
    hRap->Fill(p.Rapidity());
    hMass->Fill(p.M());
    hPhi->Fill(p.Phi());

  } //end of process
};
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCForward>(
      "upc-forward")};
}
