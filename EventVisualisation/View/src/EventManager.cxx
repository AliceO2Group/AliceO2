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
/// \file    EventManager.cxx
/// \author  Jeremi Niedziela

#include "EventVisualisationView/EventManager.h"
#include "EventVisualisationDataConverter/MinimalisticEvent.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationBase/DataSource.h"
#include "EventVisualisationBase/DataInterpreter.h"
#include "EventVisualisationBase/EventRegistration.h"
#include <EventVisualisationBase/DataSourceOffline.h>
#include <EventVisualisationDetectors/DataReaderVSD.h>

#include <TEveManager.h>
#include <TEveProjectionManager.h>
#include <TSystem.h>
#include <TEnv.h>
#include <TEveElement.h>
#include <TGListTree.h>

#include <iostream>

using namespace std;

namespace o2
{
namespace event_visualisation
{

EventManager* EventManager::instance = nullptr;

EventManager& EventManager::getInstance()
{
  if (instance == nullptr)
    instance = new EventManager();
  return *instance;
}

EventManager::EventManager() : TEveEventManager("Event", "")
{
}

void EventManager::Open()
{
  switch (mCurrentDataSourceType) {
    case SourceOnline:
      break;
    case SourceOffline: {
      DataSourceOffline* source = new DataSourceOffline();
      if (DataInterpreter::getInstance(EVisualisationGroup::VSD)) {
        DataReader* vsd = new DataReaderVSD();
        vsd->open();
        source->registerReader(vsd, EVisualisationGroup::VSD);
      }
      if (DataInterpreter::getInstance(EVisualisationGroup::RND)) {
        source->registerReader(nullptr, EVisualisationGroup::RND); // no need to read
      }
      setDataSource(source);
    } break;
    case SourceHLT:
      break;
  }
}

void EventManager::GotoEvent(Int_t no)
{
  //-1 means last event
  if (no == -1) {
    no = getDataSource()->GetEventCount() - 1;
  }
  this->currentEvent = no;
  EventRegistration::getInstance()->destroyAllEvents();
  for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; i++) {
    DataInterpreter* interpreter = DataInterpreter::getInstance((EVisualisationGroup)i);
    if (interpreter) {
      TObject* data = getDataSource()->getEventData(no, (EVisualisationGroup)i);
      TEveElement* eveElement = interpreter->interpretDataForType(data, NoData);
      EventRegistration::getInstance()->registerElement(eveElement);
    }
  }
}

void EventManager::NextEvent()
{
  Int_t event = (this->currentEvent + 1) % getDataSource()->GetEventCount();
  GotoEvent(event);
}

void EventManager::PrevEvent()
{
  GotoEvent(this->currentEvent - 1);
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

} // namespace event_visualisation
} // namespace o2
