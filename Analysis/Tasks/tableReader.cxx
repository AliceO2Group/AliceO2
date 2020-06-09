// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/ReducedInfoTables.h"
#include <TH1F.h>
#include <TMath.h>
#include <iostream>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;


struct TableReader {
  
  OutputObj<TH1F> hVtxZ{TH1F("hVtxZ", "vtx Z", 200, -20.0, 20.0)};
  OutputObj<TH1F> hVtxX{TH1F("hVtxX", "vtx X", 200, -1.0, 1.0)};
  OutputObj<TH1F> hVtxY{TH1F("hVtxY", "vtx Y", 200, -1.0, 1.0)};
  OutputObj<TH1F> hCent{TH1F("hCent", "Cent VZERO", 150, -1.0, 151.0)};
  OutputObj<TH1F> hTag{TH1F("hTag", "Tag", 100, 0.0, 100.0)};
  
  OutputObj<TH1F> hp1{TH1F("hp1", "p1", 200, -10.0, 10.0)};
  OutputObj<TH1F> hp2{TH1F("hp2", "p2", 200, -10.0, 10.0)};
  OutputObj<TH1F> hp3{TH1F("hp3", "p3", 200, -10.0, 10.0)};
  OutputObj<TH1F> hIsCartesian{TH1F("hIsCartesian", "IsCartesian", 200, -10.0, 10.0)};
  OutputObj<TH1F> hCharge{TH1F("hCharge", "Charge", 200, -10.0, 10.0)};
  
  OutputObj<TH1F> hPt{TH1F("hPt", "pt hist", 200, -10.0, 10.0)};
  OutputObj<TH1F> hEta{TH1F("hEta", "eta hist", 200, -10.0, 10.0)};
  OutputObj<TH1F> hPhi{TH1F("hPhi", "phi hist", 200, -10.0, 10.0)};
  OutputObj<TH1F> hPx{TH1F("hPx", "px hist", 200, -10.0, 10.0)};
  OutputObj<TH1F> hPy{TH1F("hPy", "py hist", 200, -10.0, 10.0)};
  OutputObj<TH1F> hPz{TH1F("hPz", "pz hist", 200, -10.0, 10.0)};
  OutputObj<TH1F> hPmom{TH1F("hPmom", "p hist", 200, -10.0, 10.0)};
  
  void init(o2::framework::InitContext&)
  {
  }

  void process(aod::ReducedEvent event, aod::ReducedTracks tracks)
  {
    /*if(event.tag()&(uint64_t(1)<<0))
      cout << "event is MB" << endl;
    else 
      cout << "event is not MB" << endl;*/
    
    hVtxX->Fill(event.vtxX());
    hVtxY->Fill(event.vtxY());
    hVtxZ->Fill(event.vtxZ());
    hCent->Fill(event.centVZERO());
    hTag->Fill(event.tag());
        
    for (auto& track : tracks) {
      
      //cout << "track pt/eta/phi  =  px/py/pz :: " << track.pt() << "/" << track.eta() << "/" << track.phi() << "  =  "
      //     << track.px() << "/" << track.py() << "/" << track.pz() << "   ########isCartesian  " << track.isCartesian() << endl;
      hp1->Fill(track.p1()); hp2->Fill(track.p2()); hp3->Fill(track.p3());
      hIsCartesian->Fill(track.isCartesian());
      hCharge->Fill(track.charge());
      
      hPt->Fill(track.pt());
      hEta->Fill(track.eta());
      hPhi->Fill(track.phi());
      hPx->Fill(track.px()); hPy->Fill(track.py()); hPz->Fill(track.pz());
      hPmom->Fill(track.pmom());
    }
    
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableReader>("table-reader")};
}
