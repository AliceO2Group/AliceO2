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
  vizSettings.firstEvent = true;

  for (int i = 0; i < NvisualisationGroups; i++) {
    vizSettings.trackVisibility[i] = true;
    vizSettings.trackColor[i] = kMagenta;
    vizSettings.trackStyle[i] = 1;
    vizSettings.trackWidth[i] = 1;
    vizSettings.clusterVisibility[i] = true;
    vizSettings.clusterColor[i] = kBlue;
    vizSettings.clusterStyle[i] = 20;
    vizSettings.clusterSize[i] = 1.0f;
  }
}

void EventManager::displayCurrentEvent()
{
  const auto multiView = MultiView::getInstance();
  const auto dataSource = getDataSource();
  if (dataSource->getEventCount() > 0) {
    if (!vizSettings.firstEvent) {
      saveVisualisationSettings();
    }

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
      multiView->registerElement(dataTypeLists[i]);
    }

    if (vizSettings.firstEvent) {
      saveVisualisationSettings();
      vizSettings.firstEvent = false;
    } else {
      restoreVisualisationSettings();
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
  struct CaloInfo {
    unsigned int index;
    std::string name;
    std::string configColor;
    int defaultColor;
    float sizeEta;
    float sizePhi;
    std::string configNoise;
    float defaultNoise;
  };

  // TODO: calculate values based on info available in O2
  const std::unordered_map<o2::dataformats::GlobalTrackID::Source, CaloInfo> caloInfos =
    {
      {o2::dataformats::GlobalTrackID::EMC, {0, "emcal", "emcal.tower.color", kYellow, 0.0143, 0.0143, "emcal.tower.noise", 0}},
      {o2::dataformats::GlobalTrackID::PHS, {1, "phos", "phos.tower.color", kYellow, 0.0046, 0.00478, "phos.tower.noise", 200}},
    };

  int size = event.getCaloCount();
  if (size > 0) {
    TEnv settings;
    ConfigurationManager::getInstance().getConfig(settings);
    const bool showAxes = settings.GetValue("axes.show", false);

    auto data = new TEveCaloDataVec(caloInfos.size());
    data->IncDenyDestroy();

    for (const auto& [det, info] : caloInfos) {
      data->RefSliceInfo(info.index).Setup(info.name.c_str(), settings.GetValue(info.configNoise.c_str(), info.defaultNoise), settings.GetValue(info.configColor.c_str(), info.defaultColor));
    }

    for (const auto& calo : event.getCalorimetersSpan()) {
      const auto& info = caloInfos.at(calo.getSource());
      const auto dEta = info.sizeEta / 2;
      const auto dPhi = info.sizePhi / 2;
      data->AddTower(calo.getEta() - dEta, calo.getEta() + dEta, calo.getPhi() - dPhi, calo.getPhi() + dPhi);
      data->FillSlice(info.index, calo.getEnergy());
    }

    data->DataChanged();
    data->SetAxisFromBins();

    const float barrelRadius = settings.GetValue("barrel.radius", 375);

    auto calo3d = new TEveCalo3D(data);
    calo3d->SetName("Calorimeters");

    calo3d->SetBarrelRadius(barrelRadius);
    calo3d->SetEndCapPos(barrelRadius);
    calo3d->SetRnrFrame(false, false); // do not draw barrel grid

    dataTypeLists[EVisualisationDataType::Calorimeters]->AddElement(calo3d);
    MultiView::getInstance()->registerElement(calo3d);
  }
}

void EventManager::saveVisualisationSettings()
{
  const auto& tracks = *dataTypeLists[EVisualisationDataType::Tracks];

  for (auto elm : tracks.RefChildren()) {
    auto trackList = static_cast<TEveTrackList*>(elm);
    int i = findGroupIndex(trackList->GetElementName());

    if (i != -1) {
      vizSettings.trackVisibility[i] = trackList->GetRnrSelf();
      vizSettings.trackColor[i] = trackList->GetLineColor();
      vizSettings.trackStyle[i] = trackList->GetLineStyle();
      vizSettings.trackWidth[i] = trackList->GetLineWidth();
    }
  }

  const auto& clusters = *dataTypeLists[EVisualisationDataType::Clusters];

  for (auto elm : clusters.RefChildren()) {
    auto clusterSet = static_cast<TEvePointSet*>(elm);
    int i = findGroupIndex(clusterSet->GetElementName());

    if (i != -1) {
      vizSettings.clusterVisibility[i] = clusterSet->GetRnrSelf();
      vizSettings.clusterColor[i] = clusterSet->GetMarkerColor();
      vizSettings.clusterStyle[i] = clusterSet->GetMarkerStyle();
      vizSettings.clusterSize[i] = clusterSet->GetMarkerSize();
    }
  }
}

void EventManager::restoreVisualisationSettings()
{
  const auto& tracks = *dataTypeLists[EVisualisationDataType::Tracks];

  for (auto elm : tracks.RefChildren()) {
    auto trackList = static_cast<TEveTrackList*>(elm);
    int i = findGroupIndex(trackList->GetElementName());

    if (i != -1) {
      const auto viz = vizSettings.trackVisibility[i];
      trackList->SetRnrSelfChildren(viz, viz);
      trackList->SetLineColor(vizSettings.trackColor[i]);
      trackList->SetLineStyle(vizSettings.trackStyle[i]);
      trackList->SetLineWidth(vizSettings.trackWidth[i]);
    }
  }

  const auto& clusters = *dataTypeLists[EVisualisationDataType::Clusters];

  for (auto elm : clusters.RefChildren()) {
    auto clusterSet = static_cast<TEvePointSet*>(elm);
    int i = findGroupIndex(clusterSet->GetElementName());

    if (i != -1) {
      const auto viz = vizSettings.clusterVisibility[i];
      clusterSet->SetRnrSelfChildren(viz, viz);
      clusterSet->SetMarkerColor(vizSettings.clusterColor[i]);
      clusterSet->SetMarkerStyle(vizSettings.clusterStyle[i]);
      clusterSet->SetMarkerSize(vizSettings.clusterSize[i]);
    }
  }
}

} // namespace event_visualisation
} // namespace o2
