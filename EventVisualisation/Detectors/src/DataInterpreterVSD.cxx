// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataInterpreterVSD.cxx
/// \brief converting VSD data to Event Visualisation primitives
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

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

namespace o2
{
namespace event_visualisation
{

DataInterpreterVSD::~DataInterpreterVSD()
{
  //this->DropEvent();
  if (mVSD) {
    delete mVSD;
    mVSD = nullptr;
  }
}

TEveElement* DataInterpreterVSD::interpretDataForType(TObject* data, EVisualisationDataType /*type*/)
{
  if (mVSD == nullptr)
    mVSD = new TEveVSD;
  this->DropEvent();

  // Connect to new event-data.
  this->mDirectory = dynamic_cast<TDirectory*>(data);
  this->mVSD->SetDirectory(this->mDirectory);

  this->AttachEvent();

  // Load event data into visualization structures.

  //        this->LoadClusters(this->fITSClusters, "ITS", 0);
  //        this->LoadClusters(this->fTPCClusters, "TPC", 1);
  //        this->LoadClusters(this->fTRDClusters, "TRD", 2);
  //        this->LoadClusters(this->fTOFClusters, "TOF", 3);

  this->LoadEsdTracks();
  return this->mTrackList;
}

void DataInterpreterVSD::LoadClusters(TEvePointSet*& ps, const TString& det_name, Int_t det_id)
{
  if (ps == nullptr) {
    ps = new TEvePointSet(det_name);
    ps->SetMainColor((Color_t)(det_id + 2));
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

void DataInterpreterVSD::AttachEvent()
{
  // Attach event data from current directory.

  mVSD->LoadTrees();
  mVSD->SetBranchAddresses();
}

void DataInterpreterVSD::DropEvent()
{
  assert(mVSD != nullptr);
  // Drop currently held event data, release current directory.
  // Drop old visualization structures.

  this->mViewers = gEve->GetViewers();
  this->mViewers->DeleteAnnotations();
  //TEveEventManager *manager = gEve->GetCurrentEvent();
  //assert(manager != nullptr);
  //manager->DestroyElements();

  // Drop old event-data.

  mVSD->DeleteTrees();
  delete mDirectory;
  mDirectory = nullptr;
}

void DataInterpreterVSD::LoadEsdTracks()
{
  // Read reconstructed tracks from current event.

  if (mTrackList == nullptr) {
    mTrackList = new TEveTrackList("ESD Tracks");
    mTrackList->SetMainColor(6);
    mTrackList->SetMarkerColor(kYellow);
    mTrackList->SetMarkerStyle(4);
    mTrackList->SetMarkerSize(0.5);
    mTrackList->SetLineWidth(1);

    mTrackList->IncDenyDestroy();
  } else {
    mTrackList->DestroyElements();
  }

  TEveTrackPropagator* trkProp = mTrackList->GetPropagator();
  // !!!! Need to store field on file !!!!
  // Can store TEveMagField ?
  trkProp->SetMagField(0.5);
  trkProp->SetStepper(TEveTrackPropagator::kRungeKutta);

  Int_t nTracks = mVSD->fTreeR->GetEntries();

  for (Int_t n = 0; n < nTracks; n++) {
    mVSD->fTreeR->GetEntry(n);

    auto* track = new TEveTrack(&mVSD->fR, trkProp);

    track->SetAttLineAttMarker(mTrackList);
    track->SetName(Form("ESD Track %d", mVSD->fR.fIndex));
    track->SetStdTitle();
    track->SetAttLineAttMarker(mTrackList);
    mTrackList->AddElement(track);
  }

  mTrackList->MakeTracks();
}

} // namespace event_visualisation
} // namespace o2