// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DataInterpreterRND.cxx
/// \author  Jeremi Niedziela

#include "EventVisualisationDetectors/DataInterpreterRND.h"

#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationBase/Track.h"
//#include "EventVisualisationView/MultiView.h"
#include "EventVisualisationDataConverter/MinimalisticEvent.h"

#include <TEveManager.h>
#include <TEveTrackPropagator.h>
#include <TGListTree.h>

#include <iostream>

using namespace std;

namespace o2  {
namespace event_visualisation {

DataInterpreterRND::DataInterpreterRND() = default;

DataInterpreterRND::~DataInterpreterRND() = default;

TEveElement* DataInterpreterRND::interpretDataForType(TObject* /*data*/, EVisualisationDataType /*type*/)
{
  int multiplicity = 500*((double)rand()/RAND_MAX)+100;
  MinimalisticEvent *minEvent = new MinimalisticEvent(15,123456,7000,multiplicity,"p-p",12736563);
  minEvent->fillWithRandomTracks();
  
  TEnv settings;
  ConfigurationManager::getInstance().getConfig(settings);
  
  const int nParticleTypes = 5;
  
  int colors[15];
  // default color scheme by type:
  colors[0] = settings.GetValue("tracks.byType.electron",600);
  colors[1] = settings.GetValue("tracks.byType.muon",416);
  colors[2] = settings.GetValue("tracks.byType.pion",632);
  colors[3] = settings.GetValue("tracks.byType.kaon",400);
  colors[4] = settings.GetValue("tracks.byType.proton",797);
  
  TEveElementList *container = new TEveElementList("Random tracks by PID");
  container->SetTitle(Form("Multiplicity = %d", minEvent->GetMultiplicity()));

  TEveTrackList *trackList[nParticleTypes];
  trackList[0] = new TEveTrackList("Electrons");
  trackList[1] = new TEveTrackList("Muons");
  trackList[2] = new TEveTrackList("Pions");
  trackList[3] = new TEveTrackList("Kaons");
  trackList[4] = new TEveTrackList("Protons");
  
  map<int, int> PIDtoListNumber;
  PIDtoListNumber[2212] = 4;
  PIDtoListNumber[-2212] = 4;
  PIDtoListNumber[321] = 3;
  PIDtoListNumber[-321] = 3;
  PIDtoListNumber[211] = 2;
  PIDtoListNumber[-211] = 2;
  PIDtoListNumber[13] = 1;
  PIDtoListNumber[-13] = 1;
  PIDtoListNumber[11] = 0;
  PIDtoListNumber[-11] = 0;
  
  const double maxR  = 520;
  const double magF  = 0.5;
  
  for (int i=0; i<nParticleTypes; i++){
    trackList[i]->GetPropagator()->SetMagField(magF);
    trackList[i]->GetPropagator()->SetMaxR(maxR);
    trackList[i]->SetMainColor(colors[i]);
    trackList[i]->SetLineWidth(settings.GetValue("tracks.width",2));
    container->AddElement(trackList[i]);
  }
  
  for (int iTrack=0; iTrack<minEvent->GetMultiplicity();iTrack++){
    MinimalisticTrack *minTrack = minEvent->getTrack(iTrack);
    
    int listNumber = PIDtoListNumber[minTrack->getPID()];
    
    Track *track = new Track();
    track->setVertex(minTrack->getVertex());
    track->setMomentum(minTrack->getMomentum());
    track->setBeta(minTrack->getBeta());
    track->SetCharge(minTrack->getCharge());
    
    track->SetPropagator(trackList[listNumber]->GetPropagator());
    track->SetAttLineAttMarker(trackList[listNumber]);
    track->SetLabel(iTrack);
    track->SetName(Form("Random track id=%d, pid=%d", iTrack, minTrack->getPID()));
    track->SetElementTitle(Form("Random track id=%d, pid=%d", iTrack, minTrack->getPID()));
    
    trackList[listNumber]->AddElement(track);
  }
  for (int i=0; i<nParticleTypes;i++){
    trackList[i]->SetName(Form("%s [%d]", trackList[i]->GetName(), trackList[i]->NumChildren()));
    trackList[i]->SetTitle(Form("N tracks=%d", trackList[i]->NumChildren()));
    trackList[i]->MakeTracks();
  }
  return container;
}
  
}
}
