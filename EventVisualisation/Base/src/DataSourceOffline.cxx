// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSourceOffline.h
/// \brief Grouping reading from file(s)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#include <EventVisualisationBase/DataSourceOffline.h>

#include <TSystem.h>
#include <TEveTreeTools.h>
#include <TEveTrack.h>
#include <TEveManager.h>
#include <TFile.h>
#include <TPRegexp.h>
#include <TEveTrackPropagator.h>
#include <TEveViewer.h>
#include <TEveEventManager.h>
#include <TEveVSD.h>
#include <TVector3.h>
#include <TObject.h>

namespace o2
{
namespace event_visualisation
{

DataReader* DataSourceOffline::instance[EVisualisationGroup::NvisualisationGroups];

TObject* DataSourceOffline::getEventData(int no, EVisualisationGroup purpose)
{
  if (instance[purpose] == nullptr)
    return nullptr;
  return instance[purpose]->getEventData(no);
}

int DataSourceOffline::GetEventCount()
{
  for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; i++) {
    if (instance[i] != nullptr)
      return instance[i]->GetEventCount();
  }
  return 1;
};

} // namespace event_visualisation
} // namespace o2
