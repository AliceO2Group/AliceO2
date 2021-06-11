// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSourceOnline.h
/// \brief Grouping reading from file(s)
/// \author julian.myrcha@cern.ch

#include <EventVisualisationBase/DataSourceOnline.h>

#include <TSystem.h>
#include <TEveTreeTools.h>
#include <TEveTrack.h>
#include <TEveManager.h>
#include <TFile.h>
#include <TPRegexp.h>
#include <TEveTrackPropagator.h>
#include <TEveVSD.h>
#include <TObject.h>

namespace o2
{
namespace event_visualisation
{

std::vector<std::pair<VisualisationEvent, std::string>> DataSourceOnline::getVisualisationList(int no)
{
  std::vector<std::pair<VisualisationEvent, std::string>> res;
  if (no < getEventCount()) {
    assert(no >= 0);
    VisualisationEvent vEvent;
    mFileWatcher.setCurrentItem(no);
    vEvent.fromFile(mFileWatcher.currentFilePath());
    res.push_back(std::make_pair(vEvent, gVisualisationGroupName[EVisualisationGroup::TPC]));
  }
  return res;
}

DataSourceOnline::DataSourceOnline(const std::string path) : mFileWatcher(path)
{
}

int DataSourceOnline::getEventCount()
{
  return this->mFileWatcher.getSize();
}

void DataSourceOnline::setCurrentEvent(Int_t currentEvent)
{
  this->mFileWatcher.setCurrentItem(currentEvent);
}

bool DataSourceOnline::refresh()
{
  return this->mFileWatcher.refresh();
}

Int_t DataSourceOnline::getCurrentEvent()
{
  return mFileWatcher.getPos();
}

} // namespace event_visualisation
} // namespace o2
