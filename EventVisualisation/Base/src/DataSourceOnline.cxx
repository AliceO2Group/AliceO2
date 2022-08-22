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
#include <TObject.h>

namespace o2
{
namespace event_visualisation
{
std::vector<std::string> DataSourceOnline::sourceFilextensions = {".json", ".root"};

std::vector<std::pair<VisualisationEvent, EVisualisationGroup>> DataSourceOnline::getVisualisationList(int no, float minTime, float maxTime, float range)
{
  std::vector<std::pair<VisualisationEvent, EVisualisationGroup>> res;
  if (getEventCount() == 2) {
    return res; // 2 means there are no real data = we have only "virtual" positions
  }
  if (no < getEventCount()) {
    assert(no >= 0);

    mFileWatcher.setCurrentItem(no);
    VisualisationEvent vEvent = this->mDataReader->getEvent(mFileWatcher.currentFilePath());

    this->setRunNumber(vEvent.getRunNumber());
    this->setFirstTForbit(vEvent.getFirstTForbit());
    this->setCollisionTime(vEvent.getCollisionTime());
    this->setTrackMask(vEvent.getTrkMask());
    this->setClusterMask(vEvent.getClMask());

    double period = vEvent.getMaxTimeOfTracks() - vEvent.getMinTimeOfTracks();
    if (period > 0) {
      this->mTimeFrameMinTrackTime = minTime * period / range + vEvent.getMinTimeOfTracks();
      this->mTimeFrameMaxTrackTime = maxTime * period / range + vEvent.getMinTimeOfTracks();
    } else {
      this->mTimeFrameMinTrackTime = vEvent.getMinTimeOfTracks();
      this->mTimeFrameMaxTrackTime = vEvent.getMaxTimeOfTracks();
    }

    for (auto filter = EVisualisationGroup::ITS;
         filter != EVisualisationGroup::NvisualisationGroups;
         filter = static_cast<EVisualisationGroup>(static_cast<int>(filter) + 1)) {
      auto filtered = VisualisationEvent(vEvent, filter, this->mTimeFrameMinTrackTime, this->mTimeFrameMaxTrackTime);
      res.push_back(std::make_pair(filtered, filter)); // we can switch on/off data
    }
  }
  return res;
}

DataSourceOnline::DataSourceOnline(const std::string path) : mFileWatcher(path, sourceFilextensions)
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

o2::detectors::DetID::mask_t DataSourceOnline::getDetectorsMask()
{
  return o2::dataformats::GlobalTrackID::getSourcesDetectorsMask(mTrackMask);
}

} // namespace event_visualisation
} // namespace o2
