// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    EventManager.cxx
/// \author  Jeremi Niedziela
/// \author  Julian Myrcha
/// \author  Michal Chwesiuk
/// \author  Piotr Nowakowski

#include "EventVisualisationView/EventManager.h"
#include "EventVisualisationView/EventManagerFrame.h"
#include "EventVisualisationView/MultiView.h"
#include "EventVisualisationView/Options.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include <EventVisualisationBase/DataSourceOnline.h>

#include <TEveManager.h>
#include <TEveTrack.h>
#include <TEveTrackPropagator.h>
#include <TEnv.h>
#include <TEveElement.h>
#include <TGListTree.h>
#include <TEveCalo.h>
#include "FairLogger.h"

#define elemof(e) (unsigned int)(sizeof(e) / sizeof(e[0]))

using namespace std;

namespace o2
{
namespace event_visualisation
{

EventManager* EventManager::instance = nullptr;

EventManager& EventManager::getInstance()
{
  if (instance == nullptr) {
    instance = new EventManager();
  }
  return *instance;
}

EventManager::EventManager() : TEveEventManager("Event", "")
{
  LOG(info) << "Initializing TEveManager";
  for (unsigned int i = 0; i < elemof(dataTypeLists); i++) {
    dataTypeLists[i] = nullptr;
  }
}

void EventManager::displayCurrentEvent()
{
  if (getDataSource()->getEventCount() > 0) {
    MultiView::getInstance()->destroyAllEvents();
    int no = getDataSource()->getCurrentEvent();

    for (int i = 0; i < EVisualisationDataType::NdataTypes; ++i) {
      dataTypeLists[i] = new TEveElementList(gDataTypeNames[i].c_str());
    }

    auto displayList = getDataSource()->getVisualisationList(no, EventManagerFrame::getInstance().getMinTimeFrameSliderValue(), EventManagerFrame::getInstance().getMaxTimeFrameSliderValue(), EventManagerFrame::MaxRange);
    for (auto it = displayList.begin(); it != displayList.end(); ++it) {
      displayVisualisationEvent(it->first, gVisualisationGroupName[it->second]);
    }

    for (int i = 0; i < EVisualisationDataType::NdataTypes; ++i) {
      MultiView::getInstance()->registerElement(dataTypeLists[i]);
    }
    // displayCalorimeters(displayList[0].first);

    MultiView::getInstance()->getAnnotation()->SetText(TString::Format("Run: %d", displayList[0].first.getRunNumber()));
  }
  MultiView::getInstance()->redraw3D();
}

void EventManager::GotoEvent(Int_t no)
{
  if (getDataSource()->getEventCount() > 0) {
    if (no == -1) {
      no = getDataSource()->getEventCount() - 1;
    }
    this->getDataSource()->setCurrentEvent(no);
    displayCurrentEvent();
  }
}

void EventManager::NextEvent()
{
  if (getDataSource()->getEventCount() > 0) {
    if (this->getDataSource()->getCurrentEvent() < getDataSource()->getEventCount() - 1) {
      Int_t event = (this->getDataSource()->getCurrentEvent() + 1) % getDataSource()->getEventCount();
      GotoEvent(event);
    }
  }
}

void EventManager::PrevEvent()
{
  if (getDataSource()->getEventCount() > 0) {
    if (this->getDataSource()->getCurrentEvent() > 0) {
      GotoEvent(this->getDataSource()->getCurrentEvent() - 1);
    }
  }
}

void EventManager::CurrentEvent()
{
  if (getDataSource()->getEventCount() > 0) {
    GotoEvent(this->getDataSource()->getCurrentEvent());
  }
}

void EventManager::Close()
{
  delete this->dataSource;
  this->dataSource = nullptr;
}

void EventManager::AfterNewEventLoaded()
{
  TEveEventManager::AfterNewEventLoaded();
}

void EventManager::AddNewEventCommand(const TString& cmd)
{
  TEveEventManager::AddNewEventCommand(cmd);
}

void EventManager::RemoveNewEventCommand(const TString& cmd)
{
  TEveEventManager::RemoveNewEventCommand(cmd);
}

void EventManager::ClearNewEventCommands()
{
  TEveEventManager::ClearNewEventCommands();
}

EventManager::~EventManager()
{
  instance = nullptr;
}

void EventManager::DropEvent()
{
  DestroyElements();
}

void EventManager::displayVisualisationEvent(VisualisationEvent& event, const std::string& detectorName)
{
  double eta = 0.1;
  size_t trackCount = event.getTrackCount();
  LOG(info) << "displayVisualisationEvent: " << trackCount << " detector: " << detectorName;
  // tracks
  auto* list = new TEveTrackList(detectorName.c_str());
  list->IncDenyDestroy();
  // clusters
  size_t clusterCount = 0;
  auto* point_list = new TEvePointSet(detectorName.c_str());
  point_list->IncDenyDestroy(); // don't delete if zero parent
  point_list->SetMarkerColor(kBlue);

  for (size_t i = 0; i < trackCount; ++i) {
    VisualisationTrack track = event.getTrack(i);
    TEveRecTrackD t;
    // double* p = track.getMomentum();
    // t.fP = {p[0], p[1], p[2]};
    t.fSign = track.getCharge() > 0 ? 1 : -1;
    auto* vistrack = new TEveTrack(&t, &TEveTrackPropagator::fgDefault);
    vistrack->SetLineColor(kMagenta);
    // vistrack->SetName(detectorName + " track: " + i);
    vistrack->SetName(track.getGIDAsString().c_str());
    size_t pointCount = track.getPointCount();
    vistrack->Reset(pointCount);

    int points = 0;
    for (size_t j = 0; j < pointCount; ++j) {
      auto point = track.getPoint(j);
      if (point[2] > eta || point[2] < -1 * eta) {
        vistrack->SetNextPoint(point[0], point[1], point[2]);
        points++;
      }
    }
    if (points > 0) {
      list->AddElement(vistrack);
    }

    // clusters connected with track
    for (size_t i = 0; i < track.getClusterCount(); ++i) {
      VisualisationCluster cluster = track.getCluster(i);
      if (cluster.Z() > eta || cluster.Z() < -1 * eta) { // temporary remove eta=0 artefacts
        point_list->SetNextPoint(cluster.X(), cluster.Y(), cluster.Z());
        clusterCount++;
      }
    }
  }

  if (trackCount != 0) {
    dataTypeLists[EVisualisationDataType::Tracks]->AddElement(list);
  }

  // global clusters (with no connection information)
  for (size_t i = 0; i < event.getClusterCount(); ++i) {
    VisualisationCluster cluster = event.getCluster(i);
    if (cluster.Z() > eta || cluster.Z() < -1 * eta) { // temporary remove eta=0 artefacts
      point_list->SetNextPoint(cluster.X(), cluster.Y(), cluster.Z());
      clusterCount++;
    }
  }

  if (clusterCount != 0) {
    dataTypeLists[EVisualisationDataType::Clusters]->AddElement(point_list);
  }

  LOG(info) << "tracks: " << trackCount << " detector: " << detectorName << ":" << dataTypeLists[EVisualisationDataType::Tracks]->NumChildren();
  LOG(info) << "clusters: " << clusterCount << " detector: " << detectorName << ":" << dataTypeLists[EVisualisationDataType::Clusters]->NumChildren();

  displayCalorimeters(event);
}

void EventManager::displayCalorimeters(VisualisationEvent& event)
{
  int size = event.getCaloCount();
  if (size > 0) {
    auto data = new TEveCaloDataVec(1);
    data->IncDenyDestroy();
    data->RefSliceInfo(0).Setup("Data", 0.3, kYellow);

    for (auto calo : event.getCalorimetersSpan()) {
      const float dX = 0.173333;
      const float dY = 0.104667;
      data->AddTower(calo.getEta(), calo.getEta() + dX, calo.getPhi(), calo.getPhi() + dY);
      data->FillSlice(0, calo.getEnergy());
    }

    data->DataChanged();
    data->SetAxisFromBins();

    float barrelRadius = 375;
    float endCalPosition = 400;

    auto calo3d = new TEveCalo3D(data);

    calo3d->SetBarrelRadius(barrelRadius);
    calo3d->SetEndCapPos(endCalPosition);

    dataTypeLists[EVisualisationDataType::Calorimeters]->AddElement(calo3d);
  }
}

} // namespace event_visualisation
} // namespace o2
