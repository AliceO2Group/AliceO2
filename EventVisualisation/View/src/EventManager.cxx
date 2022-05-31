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
#include "EventVisualisationBase/ConfigurationManager.h"
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
  const auto multiView = MultiView::getInstance();
  const auto dataSource = getDataSource();
  if (dataSource->getEventCount() > 0) {
    multiView->destroyAllEvents();
    int no = dataSource->getCurrentEvent();

    for (int i = 0; i < EVisualisationDataType::NdataTypes; ++i) {
      dataTypeLists[i] = new TEveElementList(gDataTypeNames[i].c_str());
    }

    VisualisationEvent event; // collect calorimeters in one drawing step
    auto displayList = dataSource->getVisualisationList(no, EventManagerFrame::getInstance().getMinTimeFrameSliderValue(), EventManagerFrame::getInstance().getMaxTimeFrameSliderValue(), EventManagerFrame::MaxRange);
    for (auto it = displayList.begin(); it != displayList.end(); ++it) {
      if (it->second == EVisualisationGroup::EMC || it->second == EVisualisationGroup::PHS) {
        event.appendAnotherEventCalo(it->first);
      } else {
        displayVisualisationEvent(it->first, gVisualisationGroupName[it->second]);
      }
    }
    displayCalorimeters(event);

    for (int i = 0; i < EVisualisationDataType::NdataTypes; ++i) {
      if (i != EVisualisationGroup::EMC && i != EVisualisationGroup::PHS) {
        multiView->registerElement(dataTypeLists[i]);
      }
    }
    multiView->getAnnotationTop()->SetText(TString::Format("Run %d\n%s", dataSource->getRunNumber(), dataSource->getCollisionTime().c_str()));
    auto detectors = detectors::DetID::getNames(dataSource->getDetectorsMask());
    multiView->getAnnotationBottom()->SetText(TString::Format("TFOrbit: %d\nDetectors: %s", dataSource->getFirstTForbit(), detectors.c_str()));
  }
  multiView->redraw3D();
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
    t.fSign = track.getCharge() > 0 ? 1 : -1;
    auto* vistrack = new TEveTrack(&t, &TEveTrackPropagator::fgDefault);
    vistrack->SetLineColor(kMagenta);
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
}

void EventManager::displayCalorimeters(VisualisationEvent& event)
{
  int size = event.getCaloCount();
  if (size > 0) {
    TEnv settings;
    ConfigurationManager::getInstance().getConfig(settings);
    const bool showAxes = settings.GetValue("axes.show", false);

    auto data = new TEveCaloDataVec(2); // number of detectors
    data->IncDenyDestroy();
    data->RefSliceInfo(0).Setup("emcal", 0.3, settings.GetValue("emcal.tower.color", kYellow));
    data->RefSliceInfo(1).Setup("phos", 0.3, settings.GetValue("phos.tower.color", kYellow));

    for (auto calo : event.getCalorimetersSpan()) {
      const float dX = 0.173333;
      const float dY = 0.104667; // to trzeba wziac ze stałych
      data->AddTower(calo.getEta(), calo.getEta() + dX, calo.getPhi(), calo.getPhi() + dY);
      data->FillSlice(calo.getSource() == o2::dataformats::GlobalTrackID::PHS ? 1 : 0, calo.getEnergy()); // do ktorego slice
    }

    data->DataChanged();
    data->SetAxisFromBins();

    float endCalPosition = 400;

    auto calo3d = new TEveCalo3D(data);
    calo3d->SetName("Calorimeters");

    calo3d->SetBarrelRadius(settings.GetValue("barrel.radius", 375)); // barel staring point
    calo3d->SetEndCapPos(endCalPosition);                             // scaling factor
    calo3d->SetRnrFrame(false, false);                                // do not draw barel

    dataTypeLists[EVisualisationDataType::Calorimeters]->AddElement(calo3d);
    MultiView::getInstance()->registerElement(calo3d);
  }
}

} // namespace event_visualisation
} // namespace o2
