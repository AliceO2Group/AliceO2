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
/// \file    DataInterpreterVSD.cxx
/// \author  Julian Myrcha

#include "EventVisualisationDetectors/DataInterpreterVSD.h"

#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationBase/Track.h"

#include "EventVisualisationDataConverter/MinimalisticEvent.h"
#include <EventVisualisationBase/EventRegistration.h>

#include <TEveManager.h>
#include <TEveTrackPropagator.h>
#include <TGListTree.h>

#include <iostream>

using namespace std;

namespace o2 {
namespace event_visualisation {

DataInterpreterVSD::DataInterpreterVSD() {

}


DataInterpreterVSD::~DataInterpreterVSD() {
  //this->DropEvent();
  if (fVSD) {
    delete fVSD;
    fVSD = nullptr;
  }
}

TEveElement* DataInterpreterVSD::interpretDataForType(TObject* data, EDataType type) {
  if (fVSD == nullptr)
    fVSD = new TEveVSD;
  this->DropEvent();

  // Connect to new event-data.
  this->fDirectory = dynamic_cast<TDirectory *>(data);
  this->fVSD->SetDirectory(this->fDirectory);

  this->AttachEvent();

  // Load event data into visualization structures.

//        this->LoadClusters(this->fITSClusters, "ITS", 0);
//        this->LoadClusters(this->fTPCClusters, "TPC", 1);
//        this->LoadClusters(this->fTRDClusters, "TRD", 2);
//        this->LoadClusters(this->fTOFClusters, "TOF", 3);

    this->LoadEsdTracks();
    return this->fTrackList;
}

void DataInterpreterVSD::LoadClusters(TEvePointSet *&ps, const TString &det_name, Int_t det_id) {
  if (ps == 0) {
    ps = new TEvePointSet(det_name);
    ps->SetMainColor((Color_t) (det_id + 2));
    ps->SetMarkerSize(0.5);
    ps->SetMarkerStyle(2);
    ps->IncDenyDestroy();
  } else {
    ps->Reset();
  }

  //TEvePointSelector ss(fVSD->fTreeC, ps, "fV.fX:fV.fY:fV.fZ", TString::Format("fDetId==%d", det_id));
  //ss.Select();
  ps->SetTitle(TString::Format("N=%d", ps->Size()));

  gEve->AddElement(ps);
}

void DataInterpreterVSD::AttachEvent() {
  // Attach event data from current directory.

  fVSD->LoadTrees();
  fVSD->SetBranchAddresses();
}

void DataInterpreterVSD::DropEvent() {
  assert(fVSD != nullptr);
  // Drup currently held event data, release current directory.

  // Drop old visualization structures.

  this->viewers = gEve->GetViewers();
  this->viewers->DeleteAnnotations();
  //TEveEventManager *manager = gEve->GetCurrentEvent();
  //assert(manager != nullptr);
  //manager->DestroyElements();

  // Drop old event-data.

  fVSD->DeleteTrees();
  delete fDirectory;
  fDirectory = 0;
}

void DataInterpreterVSD::LoadEsdTracks() {
  // Read reconstructed tracks from current event.

  if (fTrackList == 0) {
    fTrackList = new TEveTrackList("ESD Tracks");
    fTrackList->SetMainColor(6);
    fTrackList->SetMarkerColor(kYellow);
    fTrackList->SetMarkerStyle(4);
    fTrackList->SetMarkerSize(0.5);
    fTrackList->SetLineWidth(5);

    fTrackList->IncDenyDestroy();
  } else {
    fTrackList->DestroyElements();
    EventRegistration::getInstance()->destroyAllEvents();
  }

  TEveTrackPropagator *trkProp = fTrackList->GetPropagator();
  // !!!! Need to store field on file !!!!
  // Can store TEveMagField ?
  trkProp->SetMagField(0.5);
  trkProp->SetStepper(TEveTrackPropagator::kRungeKutta);

  Int_t nTracks = fVSD->fTreeR->GetEntries();

  for (Int_t n = 0; n < nTracks; n++) {
    fVSD->fTreeR->GetEntry(n);

    auto *track = new TEveTrack(&fVSD->fR, trkProp);
    track->SetAttLineAttMarker(fTrackList);
    track->SetName(Form("ESD Track %d", fVSD->fR.fIndex));
    track->SetStdTitle();
    track->SetAttLineAttMarker(fTrackList);
    fTrackList->AddElement(track);
  }

  fTrackList->MakeTracks();
}

}
}