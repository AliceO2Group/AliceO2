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
/// \file    EventManager.h
/// \author  Jeremi Niedziela
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_VIEW_EVENTMANAGER_H
#define ALICE_O2_EVENTVISUALISATION_VIEW_EVENTMANAGER_H

#include "EventVisualisationDataConverter/VisualisationConstants.h"
#include "EventVisualisationBase/DataReader.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "TEveCaloData.h"

#include <TEveElement.h>
#include <TEveEventManager.h>
#include <TQObject.h>

#include <string>

namespace o2
{
namespace event_visualisation
{

/// EventManager is a singleton class managing event loading.
///
/// This class is a hub for data visualisation classes, providing them with objects of requested type
/// (Raw data, hits, digits, clusters, ESDs, AODs...). It is a role of detector-specific data macros to
/// interpret data from different formats as visualisation objects (points, lines...) and register them
/// for drawing in the MultiView.

class DataSource;

class EventManager final : public TEveEventManager, public TQObject
{
 public:
  /// Returns an instance of EventManager
  static EventManager& getInstance();

  /// Setter of the current data source path
  inline void setDataSourcePath(const TString& path) { dataPath = path; }
  /// Sets the CDB path in CCDB Manager
  inline void setCdbPath(const TString& path)
  {
    ccdbApi.init(path.Data());
  }

  DataSource* getDataSource() { return dataSource; }
  void setDataSource(DataSource* dataSource) { this->dataSource = dataSource; }
  void CurrentEvent();

  void GotoEvent(Int_t /*event*/) override;
  void NextEvent() override;
  void PrevEvent() override;
  void Close() override;
  void displayCurrentEvent();

  void AfterNewEventLoaded() override;

  void AddNewEventCommand(const TString& cmd) override;
  void RemoveNewEventCommand(const TString& cmd) override;
  void ClearNewEventCommands() override;

  void DropEvent();

 private:
  struct Settings {
    bool firstEvent;
    Bool_t trackVisibility[EVisualisationGroup::NvisualisationGroups];
    Color_t trackColor[EVisualisationGroup::NvisualisationGroups];
    Style_t trackStyle[EVisualisationGroup::NvisualisationGroups];
    Width_t trackWidth[EVisualisationGroup::NvisualisationGroups];

    Bool_t clusterVisibility[EVisualisationGroup::NvisualisationGroups];
    Color_t clusterColor[EVisualisationGroup::NvisualisationGroups];
    Style_t clusterStyle[EVisualisationGroup::NvisualisationGroups];
    Size_t clusterSize[EVisualisationGroup::NvisualisationGroups];
  };

  static EventManager* instance;
  o2::ccdb::CcdbApi ccdbApi;
  TEveElementList* dataTypeLists[EVisualisationDataType::NdataTypes];
  DataSource* dataSource = nullptr;
  TString dataPath = "";
  Settings vizSettings;

  /// Default constructor
  EventManager();
  /// Default destructor
  ~EventManager() final;
  /// Deleted copy constructor
  EventManager(EventManager const&) = delete;
  /// Deleted assignemt operator
  void operator=(EventManager const&) = delete;

  void displayVisualisationEvent(VisualisationEvent& event, const std::string& detectorName);
  void displayCalorimeters(VisualisationEvent& event);
  void saveVisualisationSettings();
  void restoreVisualisationSettings();
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_VIEW_EVENTMANAGER_H
