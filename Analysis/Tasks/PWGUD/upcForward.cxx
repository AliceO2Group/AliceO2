// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include
"Framework/runDataProcessing.h"
#include
  "Framework/AnalysisTask.h"
#include
  "Framework/AnalysisDataModel.h"

#include
  "AnalysisDataModel/TrackSelectionTables.h"
#include
  "AnalysisDataModel/EventSelection.h"

#include <TH1D.h>
#include <TH2D.h>
#include
  "TLorentzVector.h"
#include
  "AnalysisCore/TriggerAliases.h" using namespace std;
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

  //Partition<aod::Muons> posMuon = o2::aod::muoncluster::charge> 0;
  //Partition<aod::Tracks> negMuon = o2::aod::muoncluster::charge< 0;
  //Partition<aod::Tracks> leftTracksPartition = aod::track::pt < 0;

  //OutputObj<TH2D> hSignalTPC1vsSignalTPC2{TH2D("hSignalTPC1vsSignalTPC2", ";TPC Signal 1;TPC Signal 2", 1000, 0., 200., 1000, 0., 200.)};

  //Filter trackFilter = (aod::track::isGlobalTrack == (uint8_t) true);

  //using MuonTracks = soa::Join<aod::ReducedMuons, aod::ReducedMuonsExtended>;
  //using ReducedEvents = soa::Join<aod::ReducedEvents, aod::ReducedEventsExtended>;

  //void process(MyEvents::iterator const& col, aod::Muons const& tracksMuon, aod::BCs::iterator const& bc)//, aod::MuonCluster const&muoncluster)
  void process(aod::BC const& bc, aod::Muons const& tracksMuon)
  //, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TrackSelection>> const& tracks)
  {
    int iSelectionCounter = 0;
    //bool checkV0 = col.bbV0A() || col.bbV0C() || col.bgV0A() || col.bgV0C(); //evgeny suggested that v0c will be fired by muon so
    hSelectionCounter->Fill(iSelectionCounter);
    iSelectionCounter++;
    hSelectionCounter->GetXaxis()->SetBinLabel(iSelectionCounter,
                                               "NoSelection");
    /*    bool checkV0 = col.bbV0A() || col.bbV0C() || col.bgV0A() || col.bgV0C();
    if (checkV0) {
      return;
    }
    hSelectionCounter->Fill(1);
    hSelectionCounter->GetXaxis()->SetBinLabel(2,"checkV0");
    bool checkFDD = col.bbFDA() || col.bbFDC() || col.bgFDA() || col.bgFDC();
    if (checkFDD) {
      return;
    }
    hSelectionCounter->Fill(2);
    hSelectionCounter->GetXaxis()->SetBinLabel(3,"checkFDD");
    //changed this  to forward trigger....
    // it is not finding the kCMUP triger but there was kMUP10 and kMUP11
    //checking available triggers
    */

    /*  for(int i; i<10;i++){
     cout <<"Trigger aliases are" <<col.alias()[i]<<endl;
   }*/

    //std::cout<<"checking triggers"<< col.alias()[kMUP11]<<"n";
    /*  if (col.alias()[kMUP10]) {
      //std::cout<<"there is no CMUP10 triggered events";
      hTriggerChecker->Fill(0);
      hTriggerChecker->GetXaxis()->SetBinLabel(1,"CMUP10");
      //return;
    }
    if (col.alias()[kINT7]) {
      //std::cout<<"there is no CMUP10 triggered events";
      hTriggerChecker->Fill(1);
      hTriggerChecker->GetXaxis()->SetBinLabel(2,"kINT7");
      //return;
    }
  if (col.alias()[kEMC7]) {
    //std::cout<<"there is no CMUP10 triggered events";
    hTriggerChecker->Fill(2);
    hTriggerChecker->GetXaxis()->SetBinLabel(3,"kEMC7");
    //return;
  }
  if (col.alias()[kINT7inMUON]) {
    //std::cout<<"there is no CMUP10 triggered events";
    hTriggerChecker->Fill(3);
    hTriggerChecker->GetXaxis()->SetBinLabel(4," kINT7inMUON");
    //return;
  }
  if (col.alias()[kMuonSingleLowPt7]) {
    //std::cout<<"there is no CMUP10 triggered events";
    hTriggerChecker->Fill(4);
    hTriggerChecker->GetXaxis()->SetBinLabel(5,"kMuonSingleLowPt7");
    //return;
  }
  if (col.alias()[kMuonUnlikeLowPt7]) {
    //std::cout<<"there is no CMUP10 triggered events";
    hTriggerChecker->Fill(5);
    hTriggerChecker->GetXaxis()->SetBinLabel(6,"kMuonUnlikeLowPt7");
    //return;
  }
  if (col.alias()[kMuonLikeLowPt7]) {
    //std::cout<<"there is no CMUP10 triggered events";
    hTriggerChecker->Fill(6);
    hTriggerChecker->GetXaxis()->SetBinLabel(7,"kMuonLikeLowPt7");
    //return;
  }
  if (col.alias()[kCUP8]) {
    //std::cout<<"there is no CMUP10 triggered events";
    hTriggerChecker->Fill(7);
    hTriggerChecker->GetXaxis()->SetBinLabel(8,"kCUP8");
    //return;
  }
*/

    /*if (col.alias()[kMUP11]) {
      hTriggerChecker->Fill(8);
      hTriggerChecker->GetXaxis()->SetBinLabel(9,"CMUP11");


      //std::cout<<"there is no CMUP11 triggered events";
      //return;
    }

    if (col.alias()[kCUP9]) {
      hTriggerChecker->Fill(9);
      hTriggerChecker->GetXaxis()->SetBinLabel(10,"CCUP9");
      //return;
    }*/

    //auto posMuon=tracksMuon.BCId();
    //auto negMuon=first+1;
    //cout<<aod::reducedtrack::charge<<endl;';
    //cout << muoncluster.track()<< endl;
    int muontracknumber = 0;
    TLorentzVector p1, p2, p;
    bool ispositive = kFALSE;
    bool isnegative = kFALSE;

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
        //cout << aod::MuonCluster::TrackId<muon><<endl;
        //cout<<"test of the iterator"<<muon.phi()<<endl;
        //cout<<"id"<<muon.bcId()<<endl;
        //auto muon1 = muon.begin()
        //auto muon2 = muon1+1

        //commenting this out now to see if I we do not need these
        /*if (muon.bcId() != col.bcId()) {
     continue;
   }*/
        //this code below is suggested by evgeny. this code is now hardcoded for run 246392 as we are not sure if trigger id is same for all the runs

        hCharge->Fill(muon.charge());

        muontracknumber++;

        if (muon.charge() > 0) {
          p1.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
          ispositive = kTRUE;
          //p1.SetXYZM(posMuon.px(), posMuon.py(), posMuon.pz(), mmuon);
        }

        if (muon.charge() < 0) {
          p2.SetXYZM(muon.px(), muon.py(), muon.pz(), mmuon);
          isnegative = kTRUE;
          //p2.SetXYZM(negMuon.px(), negMuon.py(), negMuon.pz(), mmuon);
        }
        /*  if (posMuon.Charge() * negMuon.Charge()<=0){
        cout<< "there are no opposite charge pairs"
        continue;
      }*/

        //cout<<"muontracknumber" <<muontracknumber<<endl;
        //std::cout<<"there are some CMUP11 triggered events \n";
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
      //cout<< "printing values"<< p.M()<< p1.M() <<p2.M()<<endl;
      hPt->Fill(p.Pt());
      hRap->Fill(p.Rapidity());
      hMass->Fill(p.M());
      /*  if (tracks.size() != 2) {
      return;
    }*/
      /*  auto first = tracks.begin();
    auto second = first + 1;
    if (first.charge() * second.charge() >= 0) {
      return;
    }
    //UChar_t clustermap1 = first.itsClusterMap();
    //UChar_t clustermap2 = second.itsClusterMap();
    //bool checkClusMap = TESTBIT(clustermap1, 0) && TESTBIT(clustermap1, 1) && TESTBIT(clustermap2, 0) && TESTBIT(clustermap2, 1);
  //  if (!checkClusMap) {
  //    return;
  //  }
    TLorentzVector p1, p2, p;
    p1.SetXYZM(first.px(), first.py(), first.pz(), mmuon);
    p2.SetXYZM(second.px(), second.py(), second.pz(), mmuon);
    p = p1 + p2;
    hPt->Fill(p.Pt());
    hRap->Fill(p.Rapidity());
    //float signalTPC1 = first.tpcSignal();
    //float signalTPC2 = second.tpcSignal();
    //hSignalTPC1vsSignalTPC2->Fill(signalTPC1, signalTPC2);
    //if ((p.Pt() < 0.1) && (signalTPC1 + signalTPC2 < 140.)) {
      hMass->Fill(p.M());*/
      //  }

    } // end of cmup trigger
  }   //end of process
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<UPCForward>(
      "upc-forward")};
}
