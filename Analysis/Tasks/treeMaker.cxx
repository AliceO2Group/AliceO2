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
#include "Framework/ASoAHelpers.h"
#include "Analysis/Multiplicity.h"
#include "Analysis/EventSelection.h"
#include "Analysis/Centrality.h"
#include "Analysis/ReducedEvent.h"
#include "Analysis/ReducedTrack.h"
#include <TH1F.h>
#include <TTree.h>
#include <TMath.h>




const float gkMass = 0.0005;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;


struct TreeMaker {
  
  OutputObj<TTree> fTree{"dstTree"};
  OutputObj<TH1F> vtxZ{TH1F("vtxZ", "vtx Z", 200, -20.0, 20.0)};
  
  ReducedEvent* fEvent;
  
  void init(o2::framework::InitContext&)
  {
    fTree.setObject(new TTree("DstTree","Reduced AO2D information"));
    fEvent = new ReducedEvent();
    fTree->Branch("Event",&fEvent,16000,99);
  }

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    fEvent->ClearEvent();  
    if(collision.sel7()) fEvent->SetEventTag(0);
    fEvent->SetVertex(0, collision.posX());    
    fEvent->SetVertex(1, collision.posY());    
    fEvent->SetVertex(2, collision.posZ());    
    fEvent->SetCentVZERO(collision.centV0M());
    
    vtxZ->Fill(collision.posZ());
    
    TClonesArray& reducedTracks = *(fEvent->GetTracks());
    
    for (auto& track : tracks) {
            
      ReducedTrack* reducedTrack = new (reducedTracks[fEvent->NTracks()]) ReducedTrack();
      reducedTrack->TrackId(track.globalIndex());
      reducedTrack->PtPhiEta(track.pt(), track.phi(), track.eta());
      reducedTrack->Charge(track.charge());
    }
    
    fTree->Fill();
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TreeMaker>("tree-maker")};
}
