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

#include <TEveManager.h>
#include <TEveProjectionManager.h>
#include <TSystem.h>
#include <TEnv.h>
#include <TEveElement.h>
#include <TEveTrackPropagator.h>
#include <TGListTree.h>
#include <TEveTrack.h>

using namespace std;

namespace o2  {
namespace EventVisualisation {

EventManager& EventManager::getInstance()
{
  static EventManager instance;
  return instance;
}

EventManager::EventManager() : TEveEventManager("Event",""),
mCurrentDataSourceType(SourceOffline)
{
}

EventManager::~EventManager() = default;

}
}
