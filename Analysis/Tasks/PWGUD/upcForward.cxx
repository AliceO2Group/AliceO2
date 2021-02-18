// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* To build a new task
1. you need to copy your .cxx file to ~/alice/O2/Analysis/Task/PWGUD/
2. edit ~/alice/O2/Analysis/Task/PWGUD/CMakeLists.txt... according to the format written there
3. cd to ~/alice/sw/BUILD/O2-latest/O2/Analysis/Task/PWGUD/
(it is not recommended but i found it helpul to load O2 environment by >> alienv enter O2/latest)
4. make -j n install // where n = number of task
5. if completed without error you can find your task as executable o2-analysis-<your task name you gave>
in my case it was upc-forward so it will be o2-analysis-upc-forward

I ran this code using following:
o2-analysis-timestamp --aod-file @<path to ao2d.txt> [--isPbPb] | o2-analysis-event-selection | o2-analysis-trackextension | o2-analysis-trackselection | o2-analysis-upc-forward --aod-memory-rate-limit 10000000000 --shm-segment-size 16000000000 -b
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
#include "TLorentzVector.h"
#include "AnalysisCore/TriggerAliases.h"
using namespace std;
using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
//change this from pion to muon mass
#define mmuon 0.1057
using MyEvents = soa::Join<aod::Collisions, aod::EvSels>;

struct UPCForward {
  OutputObj<TH1D> hMass{TH1D(
    "hMass",
    ";#it{m_{#mu#mu}}, GeV/c^{2};", 500, 0, 10)};
  OutputObj<TH1D> hPt{TH1D(
    "hPt",
    ";#it{p_{T}}, GeV/c;", 500, 0., 1.)};
  OutputObj<TH1D> hRap{TH1D(
    "hRap",
    ";#it{y},..;", 500, -10., 10.)};
  OutputObj<TH1D> hCharge{TH1D(
    "hCharge",
    ";#it{charge},..;", 500, -10., 10.)};
  OutputObj<TH1D> hTriggerChecker{TH1D(
    "hTriggerChecker",
    ";#it{Trigger},..;", 30, 0., 30.)};
  OutputObj<TH1D> hSelectionCounter{TH1D(
    "hSelectionCounter",
    ";#it{Selection},..;", 30, 0., 30.)};


  void process(aod::BC const& bc, aod::Muons const& tracksMuon)

  {
    int iSelectionCounter = 0;

    hSelectionCounter->Fill(iSelectionCounter);
    iSelectionCounter++;
    hSelectionCounter->GetXaxis()->SetBinLabel(iSelectionCounter,
                                               "NoSelection");

    int muontracknumber = 0;
    TLorentzVector p1, p2, p;
    bool ispositive = kFALSE;
    bool isnegative = kFALSE;
    /*this code below is suggested by evgeny.
    this code is now hardcoded for run 246392 as we are not sure if trigger id is same for all the runs*/
    uint64_t classIndexMUP11 = 54; // 246392
    bool isMUP11fired = bc.triggerMask() & (1ull << classIndexMUP11);
    if (isMUP11fired) {
      LOGF(info,
           "mup11 fired");
      hTriggerChecker->Fill(8);
      hTriggerChecker->GetXaxis()->SetBinLabel(9,
                                               "CMUP11_New");
      hSelectionCounter->Fill(iSelectionCounter);
      iSelectionCounter++;
      hSelectionCounter->GetXaxis()->SetBinLabel(iSelectionCounter,
                                                 "CMup11Trigger");

      for (auto& muon : tracksMuon) {


        hCharge->Fill(muon.charge());

        muontracknumber++;

        if (muon.charge() > 0) {
          p1.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
          ispositive = kTRUE;

        }

        if (muon.charge() < 0) {
          p2.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
          isnegative = kTRUE;

        }

      }
      if (muontracknumber != 2) {
        return;
      }
      hSelectionCounter->Fill(iSelectionCounter);
      iSelectionCounter++;
      hSelectionCounter->GetXaxis()->SetBinLabel(iSelectionCounter,
                                                 "twotracks");

      if (!ispositive || !isnegative) {
        return;
      }

      hSelectionCounter->Fill(iSelectionCounter);
      iSelectionCounter++;
      hSelectionCounter->GetXaxis()->SetBinLabel(iSelectionCounter,
                                                 "oppositecharge");
      p = p1 + p2;

      hPt->Fill(p.Pt());
      hRap->Fill(p.Rapidity());
      hMass->Fill(p.M());


    } // end of cmup trigger
  }   //end of process
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCForward>(
      "upc-forward")};
}
