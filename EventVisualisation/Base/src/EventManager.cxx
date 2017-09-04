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
/// \file    EveInitializer.cxx
/// \author  Jeremi Niedziela

#include "EventManager.h"

#include "MultiView.h"

#include <TEveManager.h>
#include <TEveProjectionManager.h>
#include <TSystem.h>

using namespace std;

namespace o2  {
namespace EventVisualisation {

EventManager* EventManager::sMaster  = nullptr;

EventManager* EventManager::getInstance()
{
  if(!sMaster){
    new EventManager();
  }
  return sMaster;
}

void EventManager::registerEvent(TEveElement* event)
{
  auto multiView = MultiView::getInstance();
  
  gEve->AddElement(event,multiView->getScene(MultiView::Scene3dEvent));
  multiView->getProjection(MultiView::ProjectionRphi)->ImportElements(event,multiView->getScene(MultiView::SceneRphiEvent));
  multiView->getProjection(MultiView::ProjectionZrho)->ImportElements(event,multiView->getScene(MultiView::SceneZrhoEvent));
}

void EventManager::restroyAllEvents()
{
  auto multiView = MultiView::getInstance();
  
  multiView->getScene(MultiView::Scene3dEvent)->DestroyElements();
  multiView->getScene(MultiView::SceneRphiEvent)->DestroyElements();
  multiView->getScene(MultiView::SceneZrhoEvent)->DestroyElements();
}

EventManager::EventManager() : TEveEventManager("Event",""),
mCurrentDataSourceType(SourceOffline)
{
  sMaster = this;
}

EventManager::~EventManager()
{
}
  
}
}
