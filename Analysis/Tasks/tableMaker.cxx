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
#include "Analysis/ReducedInfoTables.h"
#include <TH1F.h>
#include <TMath.h>
#include <iostream>

using std::cout;
using std::endl;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;


struct TableMaker {

  Produces<aod::ReducedEvents> event;
  Produces<aod::ReducedEventsExtended> eventExtended;
  Produces<aod::ReducedEventsVtxCov> eventVtxCov;
  Produces<aod::ReducedTracks> trackBasic;
  Produces<aod::ReducedTracksBarrel> trackBarrel;
  //Produces<aod::ReducedTracksMuon> trackMuon;
  
  OutputObj<TH1F> vtxZ{TH1F("vtxZ", "vtx Z", 200, -20.0, 20.0)};
  OutputObj<TH1F> vtxX{TH1F("vtxX", "vtx X", 2000, -1.0, 1.0)};
  OutputObj<TH1F> vtxY{TH1F("vtxY", "vtx Y", 2000, -1.0, 1.0)};
  
  void init(o2::framework::InitContext&)
  {
  }

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator collision, /*aod::Muons const& tracksMuon,*/ aod::BCs const& bcs, soa::Join<aod::Tracks, aod::TracksExtra> const& tracksBarrel)
  {
    uint64_t tag = 0;
    if(collision.sel7()) tag |= (uint64_t(1)<<0);
    
    vtxZ->Fill(collision.posZ());
    vtxX->Fill(collision.posX());
    vtxY->Fill(collision.posY());
    
    event(tag, collision.bc().runNumber(), collision.posX(), collision.posY(), collision.posZ(), collision.numContrib());
    eventExtended(collision.bc().globalBC(), collision.bc().triggerMask(), collision.collisionTime(), collision.centV0M());
    eventVtxCov(collision.covXX(), collision.covXY(), collision.covXZ(), collision.covYY(), collision.covYZ(), collision.covZZ(), collision.chi2());
    
    for (auto& track : tracksBarrel) {
      
      if(track.pt()<1.0) continue;
            
      trackBasic(collision, track.globalIndex(), track.pt(), track.phi(), track.eta(), short(0), track.charge(), uint64_t(0));
      trackBarrel(track.tpcInnerParam(), track.flags(), track.itsClusterMap(), track.itsChi2NCl(), 
                  track.tpcNClsFound(), track.tpcNClsFindable(), track.tpcNClsCrossedRows(), track.tpcNClsShared(), track.tpcChi2NCl(), track.tpcSignal(),
                  track.trdNTracklets(), track.trdChi2(), track.trdSignal(),
                  track.tofSignal(), track.tofChi2(), track.length());
    }
    
    /*for (auto& muon : tracksMuon) {
      // TODO: add proper information for muon tracks
      trackBasic(collision, 0, muon.inverseBendingMomentum(), muon.thetaX(), muon.thetaY(), short(0), short(0), uint64_t(1));
      trackMuon(muon.chi2(), muon.chi2MatchTrigger());
    }*/
    
  }
  
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableMaker>("table-maker")};
}
