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



#include "EventVisualisationBase/EventManager.h"
#include "EventVisualisationDataConverter/MinimalisticEvent.h"
#include "EventVisualisationBase/Track.h"
#include "EventVisualisationBase/ConfigurationManager.h"
#include "EventVisualisationBase/DataSource.h"
#include "EventVisualisationBase/DataInterpreter.h"
#include "EventVisualisationBase/EventRegistration.h"

#include <TEveManager.h>
#include <TEveProjectionManager.h>
#include <TSystem.h>
#include <TEnv.h>
#include <TEveElement.h>
#include <TEveTrackPropagator.h>
#include <TGListTree.h>
#include <TEveTrack.h>
#include <iostream>
#include <EventVisualisationBase/DataSourceOffline.h>
#include <EventVisualisationBase/DataReaderVSD.h>


using namespace std;

namespace o2  {
namespace event_visualisation {

EventManager *EventManager::instance = nullptr;

EventManager& EventManager::getInstance()
{
  if( instance == nullptr)
      instance = new EventManager();
  return *instance;
}

EventManager::EventManager() : TEveEventManager("Event", "") {
    std::cout << "EventManager::EventManager()" << std::endl;
}

void EventManager::Open() {
    std::cout << "EventManager::Open()" << std::endl;

    switch(mCurrentDataSourceType)
    {
        case SourceOnline:
            break;
        case SourceOffline: {
              DataSourceOffline *source = new DataSourceOffline();
              if(DataInterpreter::getInstance(EVisualisationGroup::VSD)) {
                DataReader *vsd = new DataReaderVSD();
                vsd->open();
                source->registerReader(vsd, EVisualisationGroup::VSD);
              }
              if(DataInterpreter::getInstance(EVisualisationGroup::RND)) {
                source->registerReader(nullptr, EVisualisationGroup::RND);  // no need to read
              }
              setDataSource(source);
            }
            break;
        case SourceHLT:
            break;
    }
    //TEveEventManager::Open();
}

void EventManager::GotoEvent(Int_t no) {
    std::cout << "EventManager::GotoEvent("<<no<<")" << std::endl;
    //-1 means last event
    if(no == -1) {
        no = getDataSource()->GetEventCount()-1;
    }
    this->currentEvent = no;
    EventRegistration::getInstance()->destroyAllEvents();
    for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; i++) {
      DataInterpreter* interpreter = DataInterpreter::getInstance((EVisualisationGroup)i);
      if(interpreter) {
        TObject *data = getDataSource()->getEventData(no, (EVisualisationGroup)i);
        TEveElement *eveElement = interpreter->interpretDataForType(data, NoData);
        EventRegistration::getInstance()->registerElement(eveElement);
      }
    }
}

void EventManager::NextEvent() {
    std::cout << "EventManager::NextEvent()" << std::endl;
    Int_t event = (this->currentEvent + 1) % getDataSource()->GetEventCount();
    GotoEvent(event);
}

void EventManager::PrevEvent() {
    std::cout << "EventManager::PrevEvent()" << std::endl;
    GotoEvent(this->currentEvent - 1);
}

void EventManager::Close() {
    std::cout << "EventManager::Close()" << std::endl;
    delete this->dataSource;
    this->dataSource = nullptr;
}

void EventManager::AfterNewEventLoaded() {
    std::cout << "EventManager::AfterNewEventLoaded()" << std::endl;
    TEveEventManager::AfterNewEventLoaded();
}

void EventManager::AddNewEventCommand(const TString &cmd) {
    std::cout << "EventManager::AddNewEventCommand" << std::endl;
    TEveEventManager::AddNewEventCommand(cmd);
}

void EventManager::RemoveNewEventCommand(const TString &cmd) {
    std::cout << "EventManager::RemoveNewEventCommand" << std::endl;
    TEveEventManager::RemoveNewEventCommand(cmd);
}

void EventManager::ClearNewEventCommands() {
    std::cout << "EventManager::ClearNewEventCommands()" << std::endl;
    TEveEventManager::ClearNewEventCommands();
}

EventManager::~EventManager() {
    std::cout << "EventManager::~EventManager()" << std::endl;
    instance = nullptr;
}

void EventManager::DropEvent() {
  DestroyElements();
}


}
}

