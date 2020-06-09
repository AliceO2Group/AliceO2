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


using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;


struct TableMaker {

  Produces<aod::ReducedEvents> eventTable;
  //Produces<aod::Collisions> col;
  Produces<aod::ReducedTracks> trackTable;
  
  OutputObj<TH1F> vtxZ{TH1F("vtxZ", "vtx Z", 200, -20.0, 20.0)};
  OutputObj<TH1F> vtxX{TH1F("vtxX", "vtx X", 2000, -1.0, 1.0)};
  OutputObj<TH1F> vtxY{TH1F("vtxY", "vtx Y", 2000, -1.0, 1.0)};
  
  void init(o2::framework::InitContext&)
  {
  }

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    uint64_t tag = 0;
    if(collision.sel7()) tag |= (uint64_t(1)<<0);
    float vtx[3] = {collision.posX(), collision.posY(), collision.posZ()};
    float centVZERO = collision.centV0M();
    vtxZ->Fill(collision.posZ());
    vtxX->Fill(collision.posX());
    vtxY->Fill(collision.posY());
    eventTable(tag, vtx[0], vtx[1], vtx[2], centVZERO);
    
    for (auto& track : tracks) {
      uint16_t trackId = track.globalIndex();
      float p[3] = {track.pt(), track.phi(), track.eta()};
      uint8_t isCartesian = 0;
      short charge = track.charge();
      uint64_t flags = 0;
      if(p[0]<1.0) continue;
      if(TMath::Abs(p[2])>0.9) continue;
      trackTable(collision,trackId,p[0],p[1],p[2],isCartesian,charge, flags);
    }
    
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TableMaker>("table-maker")};
}
