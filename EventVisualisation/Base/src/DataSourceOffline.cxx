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

#define elemof(e) (unsigned int)(sizeof(e) / sizeof(e[0]))

namespace o2
{
namespace event_visualisation
{

VisualisationEvent DataSourceOffline::getEventData(int no, EVisualisationGroup purpose, EVisualisationDataType dataType)
{
  if (mDataReaders[purpose] == nullptr) {
    return VisualisationEvent({.eventNumber = -1,
                               .runNumber = -1,
                               .energy = -1,
                               .multiplicity = -1,
                               .collidingSystem = "",
                               .timeStamp = 0});
  }
  return mDataReaders[purpose]->getEvent(no, dataType);
}

int DataSourceOffline::getEventCount()
{
  for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; i++) {
    if (mDataReaders[i] != nullptr) {
      return mDataReaders[i]->GetEventCount();
    }
  }
  return 1;
};

void DataSourceOffline::setCurrentEvent(Int_t currentEvent)
{
  this->mCurrentEvent = currentEvent;
}

std::vector<std::pair<VisualisationEvent, std::string>> DataSourceOffline::getVisualisationList(int no)
{
  std::vector<std::pair<VisualisationEvent, std::string>> res;
  for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; ++i) {
    DataReader* reader = mDataReaders[i];
    if (reader) {
      for (int dataType = 0; dataType < EVisualisationDataType::NdataTypes; ++dataType) {
        VisualisationEvent event = getEventData(no, (EVisualisationGroup)i, (EVisualisationDataType)dataType);
        res.push_back(std::make_pair(event, gVisualisationGroupName[i]));
      }
    }
  }

  return res;
}

DataSourceOffline::DataSourceOffline()
{
  for (unsigned int i = 0; i < elemof(mDataReaders); i++) {
    mDataReaders[i] = nullptr;
  }
  for (int i = 0; i < EVisualisationGroup::NvisualisationGroups; i++) {
    if (mDataReaders[i] != nullptr) {
      mDataReaders[i]->open();
      this->registerReader(mDataReaders[i], static_cast<EVisualisationGroup>(i));
    }
  }
}

void DataSourceOffline::registerReader(DataReader* reader, EVisualisationGroup type)
{
  mDataReaders[type] = reader;
}

Int_t DataSourceOffline::getCurrentEvent()
{
  return this->mCurrentEvent;
}

} // namespace event_visualisation
} // namespace o2
