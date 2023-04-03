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

#include <unordered_map>
#include "EventVisualisationView/EventManager.h"
#include "EventVisualisationView/EventManagerFrame.h"
#include "EventVisualisationView/MultiView.h"
#include "EventVisualisationView/Options.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"
#include <EventVisualisationBase/DataSourceOnline.h>
#include "EventVisualisationBase/ConfigurationManager.h"
#include "DataFormatsParameters/ECSDataAdapters.h"
#include <TEveManager.h>
#include <TEveTrack.h>
#include <TEveTrackPropagator.h>
#include <TEnv.h>
#include <TEveElement.h>
#include <TGListTree.h>
#include <TEveCalo.h>
#include <fairlogger/Logger.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <gsl/span>

#define elemof(e) (unsigned int)(sizeof(e) / sizeof(e[0]))

using namespace std;
using namespace rapidjson;

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

  ConfigurationManager::getInstance().getConfig(settings);

  vizSettings.firstEvent = true;

  for (int i = 0; i < NvisualisationGroups; i++) {
    vizSettings.trackVisibility[i] = true;
    vizSettings.trackColor[i] = settings.GetValue("tracks.byType.unknown", kMagenta);
    vizSettings.trackStyle[i] = 1;
    vizSettings.trackWidth[i] = 1;
    vizSettings.clusterVisibility[i] = true;
    vizSettings.clusterColor[i] = settings.GetValue("clusters.byType.unknown", kBlue);
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
      dataTypeListsPhi[i] = new TEveElementList(gDataTypeNames[i].c_str());
    }

    VisualisationEvent event; // collect calorimeters in one drawing step
    auto displayList = dataSource->getVisualisationList(no, EventManagerFrame::getInstance().getMinTimeFrameSliderValue(), EventManagerFrame::getInstance().getMaxTimeFrameSliderValue(), EventManagerFrame::MaxRange);

    for (auto it = displayList.begin(); it != displayList.end(); ++it) {
      if (it->second == EVisualisationGroup::EMC || it->second == EVisualisationGroup::PHS) {
        displayCalorimeters(it->first, gVisualisationGroupName[it->second]);
      } else {
        displayVisualisationEvent(it->first, gVisualisationGroupName[it->second]);
      }
    }
    multiView->registerElements(dataTypeLists, dataTypeListsPhi);

    if (vizSettings.firstEvent) {
      ifstream s(TEMP_SETTINGS_PATH);
      if (s.good()) {
        restoreVisualisationSettings();
      } else {
        saveVisualisationSettings();
      }
      vizSettings.firstEvent = false;
    } else {
      restoreVisualisationSettings();
    }

    if (dataSource->getRunNumber() != -1) {
      if (this->mShowDate) {
        multiView->getAnnotationTop()->SetText(
          TString::Format("Run %d %s\n%s", dataSource->getRunNumber(),
                          std::string(parameters::GRPECS::RunTypeNames[dataSource->getRunType()]).c_str(),
                          dataSource->getFileTime().c_str()));
      } else {
        multiView->getAnnotationTop()->SetText(TString::Format("Run %d", dataSource->getRunNumber()));
      }
      auto detectors = detectors::DetID::getNames(dataSource->getDetectorsMask());
      multiView->getAnnotationBottom()->SetText(
        TString::Format("TFOrbit: %d\nDetectors: %s", dataSource->getFirstTForbit(), detectors.c_str()));
    } else {
      multiView->getAnnotationTop()->SetText("No Available Data to Display");
    }
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
  point_list->SetMarkerColor(settings.GetValue("clusters.byType.unknown", kBlue));

  for (size_t i = 0; i < trackCount; ++i) {
    VisualisationTrack track = event.getTrack(i);
    TEveRecTrackD t;
    t.fSign = track.getCharge() > 0 ? 1 : -1;
    auto* vistrack = new TEveTrack(&t, &TEveTrackPropagator::fgDefault);
    vistrack->SetLineColor(settings.GetValue("tracks.byType.unknown", kMagenta));
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
    if (detectorName != "MCH" && detectorName != "MFT" && detectorName != "MID") {
      // LOG(info) << "phi: " << trackCount << " detector: " << detectorName;
      dataTypeListsPhi[EVisualisationDataType::Tracks]->AddElement(list);
    }
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
    if (detectorName != "MCH" && detectorName != "MFT" && detectorName != "MID") {
      // LOG(info) << "phi: " << clusterCount << " detector: " << detectorName;
      dataTypeListsPhi[EVisualisationDataType::Clusters]->AddElement(point_list);
    }
  }

  LOG(info) << "tracks: " << trackCount << " detector: " << detectorName << ":" << dataTypeLists[EVisualisationDataType::Tracks]->NumChildren();
  LOG(info) << "clusters: " << clusterCount << " detector: " << detectorName << ":" << dataTypeLists[EVisualisationDataType::Clusters]->NumChildren();
}

void EventManager::displayCalorimeters(VisualisationEvent& event, const std::string& detectorName)
{
  if (event.getCaloCount() > 0) {
    struct CaloInfo {
      std::string name;
      std::string configColor;
      int defaultColor;
      std::string configSizeEta;
      float sizeEta;
      std::string configSizePhi;
      float sizePhi;
      std::string configNoise;
      float defaultNoise;
      std::string configTransparency;
      int defaultTransparency;
      std::string configBarrelRadius;
      int defaultBarrelRadius;
      std::string configTowerMaxHeight;
      float defaultTowerMaxHeight;
      std::string configMaxValAbs;
      float defaultMaxValAbs;
    };

    // TODO: calculate values based on info available in O2
    static const std::unordered_map<o2::dataformats::GlobalTrackID::Source, CaloInfo> caloInfos =
      {
        {o2::dataformats::GlobalTrackID::EMC, {"emcal", "emcal.tower.color", kYellow, "emcal.tower.size.eta", 0.0143, "emcal.tower.size.phi", 0.0143, "emcal.tower.noise", 0, "emcal.tower.transparency", 101, "emcal.barrel.radius", 500, "emcal.tower.max.height", 80, "emcal.tower.max.val.abs", 100}},
        {o2::dataformats::GlobalTrackID::PHS, {"phos", "phos.tower.color", kYellow, "phos.tower.size.eta", 0.0046, "phos.tower.size.phi", 0.00478, "phos.tower.noise", 200, "phos.tower.transparency", 101, "phos.barrel.radius", 550, "phos.tower.max.height", 80, "phos.tower.max.val.abs", 100}},
      };

    auto data = new TEveCaloDataVec(1);
    data->IncDenyDestroy();
    auto key = detectorName == "EMC" ? o2::dataformats::GlobalTrackID::EMC : o2::dataformats::GlobalTrackID::PHS;

    const CaloInfo& info = caloInfos.at(key);
    data->RefSliceInfo(0).Setup(info.name.c_str(),
                                settings.GetValue(info.configNoise.c_str(), info.defaultNoise),
                                settings.GetValue(info.configColor.c_str(), info.defaultColor),
                                settings.GetValue(info.configTransparency.c_str(), info.defaultTransparency));

    const auto dEta = settings.GetValue(info.configSizeEta.c_str(), info.sizeEta) / 2.0;
    const auto dPhi = settings.GetValue(info.configSizePhi.c_str(), info.sizePhi) / 2.0;
    const float barrelRadius = settings.GetValue(info.configBarrelRadius.c_str(), info.defaultBarrelRadius);

    struct pair_hash {
      std::size_t operator()(const std::pair<float, float>& pair) const
      {
        return std::hash<float>()(pair.first + 1000.0 * pair.second);
      }
    };
    std::unordered_map<std::pair<float, float>, float, pair_hash> map; // sum up entries for the same tower
    for (const auto& calo : event.getCalorimetersSpan()) {
      map[std::make_pair(calo.getEta(), calo.getPhi())] += calo.getEnergy();
    }

    for (const auto& entry : map) {
      auto [eta, phi] = entry.first;
      data->AddTower(eta - dEta, eta + dEta, phi - dPhi, phi + dPhi);
      data->FillSlice(0, entry.second);
    }

    // remove artefacts
    data->AddTower(-0.5, 0.5, -1.574 - 0.1, -1.574 + 0.1);
    data->AddTower(-0.5, 0.5, 1.574 - 0.1, 1.574 + 0.1);
    data->AddTower(-0.5, 0.5, -0.593 - 0.1, -0.593 + 0.1);
    data->AddTower(-0.5, 0.5, -0.726 - 0.1, -0.726 + 0.1);
    data->AddTower(-0.5, 0.5, -3.028 - 0.1, -3.028 + 0.1);
    data->AddTower(-0.5, 0.5, -1.915 - 0.1, -1.915 + 0.1);

    data->DataChanged();
    data->SetAxisFromBins();

    auto calo3d = new TEveCalo3D(data);
    calo3d->SetName(detectorName.c_str());
    calo3d->SetScaleAbs(kTRUE);
    calo3d->SetMaxTowerH(settings.GetValue(info.configTowerMaxHeight.c_str(), info.defaultTowerMaxHeight));
    calo3d->SetMaxValAbs(settings.GetValue(info.configMaxValAbs.c_str(), info.defaultMaxValAbs));
    // calo3d->SetAutoRange(kTRUE);

    calo3d->SetBarrelRadius(barrelRadius);
    calo3d->SetEndCapPos(barrelRadius);
    calo3d->SetRnrFrame(false, false); // do not draw barrel grid

    dataTypeLists[EVisualisationDataType::Calorimeters]->AddElement(calo3d);
    dataTypeListsPhi[EVisualisationDataType::Calorimeters]->AddElement(calo3d);
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

  ofstream settings(TEMP_SETTINGS_PATH);

  if (settings.good()) {
    Document d;
    d.SetObject();
    auto& allocator = d.GetAllocator();

    auto jsonArray = [](const auto& array, auto& allocator) {
      Value arr(kArrayType);

      for (const auto& value : array) {
        arr.PushBack(value, allocator);
      }

      return arr;
    };

    d.AddMember("trackVisibility", jsonArray(vizSettings.trackVisibility, allocator), allocator);
    d.AddMember("trackColor", jsonArray(vizSettings.trackColor, allocator), allocator);
    d.AddMember("trackStyle", jsonArray(vizSettings.trackStyle, allocator), allocator);
    d.AddMember("trackWidth", jsonArray(vizSettings.trackWidth, allocator), allocator);
    d.AddMember("clusterVisibility", jsonArray(vizSettings.clusterVisibility, allocator), allocator);
    d.AddMember("clusterColor", jsonArray(vizSettings.clusterColor, allocator), allocator);
    d.AddMember("clusterStyle", jsonArray(vizSettings.clusterStyle, allocator), allocator);
    d.AddMember("clusterSize", jsonArray(vizSettings.clusterSize, allocator), allocator);

    auto jsonCamera = [&jsonArray](MultiView::EViews view, auto& allocator) {
      Value obj(kObjectType);

      auto& camera = MultiView::getInstance()->getView(view)->GetGLViewer()->CurrentCamera();

      const gsl::span baseSpan(camera.RefCamBase().CArr(), 16);
      obj.AddMember("base", jsonArray(baseSpan, allocator), allocator);

      const gsl::span transSpan(camera.GetCamTrans().CArr(), 16);
      obj.AddMember("trans", jsonArray(transSpan, allocator), allocator);

      if (camera.IsOrthographic()) {
        obj.AddMember("zoom", dynamic_cast<TGLOrthoCamera&>(camera).GetZoom(), allocator);
      } else if (camera.IsPerspective()) {
        obj.AddMember("fov", dynamic_cast<TGLPerspectiveCamera&>(camera).GetFOV(), allocator);
      }

      return obj;
    };

    d.AddMember("camera3d", jsonCamera(MultiView::View3d, allocator), allocator);
    d.AddMember("cameraRphi", jsonCamera(MultiView::ViewRphi, allocator), allocator);
    d.AddMember("cameraZY", jsonCamera(MultiView::ViewZY, allocator), allocator);

    StringBuffer strbuf;
    Writer<StringBuffer> writer(strbuf);
    d.Accept(writer);

    settings << strbuf.GetString();
  }
}

void EventManager::restoreVisualisationSettings()
{
  ifstream settings(TEMP_SETTINGS_PATH);

  if (settings.good()) {
    string json((istreambuf_iterator<char>(settings)), istreambuf_iterator<char>());
    Document d;
    d.Parse(json.c_str());

    auto updateArray = [](auto& array, const auto& document, const char* name, const auto& accessor) {
      for (size_t i = 0; i < elemof(array); ++i) {
        array[i] = accessor(document[name][i]);
      }
    };

    auto getBool = [](const GenericValue<UTF8<char>>& v) { return v.GetBool(); };
    auto getUint = [](const GenericValue<UTF8<char>>& v) { return v.GetUint(); };
    auto getFloat = [](const GenericValue<UTF8<char>>& v) { return v.GetFloat(); };

    updateArray(vizSettings.trackVisibility, d, "trackVisibility", getBool);
    updateArray(vizSettings.trackColor, d, "trackColor", getUint);
    updateArray(vizSettings.trackStyle, d, "trackStyle", getUint);
    updateArray(vizSettings.trackWidth, d, "trackWidth", getUint);
    updateArray(vizSettings.clusterVisibility, d, "clusterVisibility", getBool);
    updateArray(vizSettings.clusterColor, d, "clusterColor", getUint);
    updateArray(vizSettings.clusterStyle, d, "clusterStyle", getUint);
    updateArray(vizSettings.clusterSize, d, "clusterSize", getFloat);

    auto updateCamera = [getFloat](MultiView::EViews view, const auto& document, const char* name) {
      auto& camera = MultiView::getInstance()->getView(view)->GetGLViewer()->CurrentCamera();

      std::array<Double_t, 16> values;

      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = getFloat(document[name]["base"][i]);
      }
      camera.RefCamBase() = TGLMatrix(values.data());

      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = getFloat(document[name]["trans"][i]);
      }
      camera.RefCamTrans() = TGLMatrix(values.data());

      if (camera.IsOrthographic()) {
        dynamic_cast<TGLOrthoCamera&>(camera).SetZoom(getFloat(document[name]["zoom"]));
      } else if (camera.IsPerspective()) {
        dynamic_cast<TGLPerspectiveCamera&>(camera).SetFOV(getFloat(document[name]["fov"]));
      }

      camera.IncTimeStamp();
    };

    updateCamera(MultiView::View3d, d, "camera3d");
    updateCamera(MultiView::ViewRphi, d, "cameraRphi");
    updateCamera(MultiView::ViewZY, d, "cameraZY");
  }

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

  MultiView::getInstance()->redraw3D();
}

} // namespace event_visualisation
} // namespace o2
