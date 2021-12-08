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

/// \file DataSourceOffline.cxx
/// \brief Grouping reading from file(s)
/// \author p.nowakowski@cern.ch

#include <EventVisualisationBase/DataSourceOffline.h>

#include <TSystem.h>
#include <TEveTreeTools.h>
#include <TEveTrack.h>
#include <TEveManager.h>
#include <TFile.h>
#include <TPRegexp.h>
#include <TObject.h>
#include <fmt/core.h>
#include <FairLogger.h>

namespace o2
{
namespace event_visualisation
{
DataSourceOffline::DataSourceOffline() : mCurrentEvent(0)
{
}

void DataSourceOffline::setCurrentEvent(Int_t currentEvent)
{
  mCurrentEvent = currentEvent;
}

std::vector<std::pair<VisualisationEvent, EVisualisationGroup>> DataSourceOffline::getVisualisationList(int no)
{
  std::vector<std::pair<VisualisationEvent, EVisualisationGroup>> res;

  if (no < getEventCount()) {
    assert(no >= 0);

    VisualisationEvent vEvent = mEvents.at(mCurrentEvent);

    for(auto filter = EVisualisationGroup::ITS;
         filter != EVisualisationGroup::NvisualisationGroups;
         filter = static_cast<EVisualisationGroup>(static_cast<int>(filter) + 1)) {
      auto filtered = VisualisationEvent(vEvent, filter);
      res.push_back(std::make_pair(filtered, filter));  // we can switch on/off data
    }
  }

  return res;
}

void DataSourceOffline::addEvent(VisualisationEvent const& event)
{
  mEvents.push_back(event);
}

} // namespace event_visualisation
} // namespace o2
