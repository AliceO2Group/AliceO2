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

/// \file DataReaderITS.h
/// \brief reading from file(s)
/// \author julian.myrcha@cern.ch
/// \author p.nowakowski@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
#define ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H

#include <EventVisualisationDataConverter/VisualisationConstants.h>
#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationDataConverter/VisualisationEvent.h>
#include <utility>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSource
{
 protected:
  DataReader* mDataReader = nullptr;
  float mTimeFrameMinTrackTime = 0;
  float mTimeFrameMaxTrackTime = 0;

 public:
  float getTimeFrameMinTrackTime() const
  {
    return mTimeFrameMinTrackTime;
  }

  float getTimeFrameMaxTrackTime() const
  {
    return mTimeFrameMaxTrackTime;
  }

 public:
  void registerReader(DataReader* reader) { this->mDataReader = reader; }
  virtual Int_t getCurrentEvent() { return 0; };
  virtual void setCurrentEvent(Int_t /*currentEvent*/){};
  virtual int getEventCount() { return 0; };
  virtual bool refresh() { return false; }; // recompute
  DataSource() = default;

  /// Default destructor
  virtual ~DataSource() = default;

  /// Deleted copy constructor
  DataSource(DataSource const&) = delete;

  /// Deleted assignemt operator
  void operator=(DataSource const&) = delete;

  virtual std::vector<std::pair<VisualisationEvent, EVisualisationGroup>> getVisualisationList(int no, float minTime, float maxTime, float range) = 0;
  virtual void rollToNext(){};
  virtual void changeDataFolder(std::string /*newFolder*/){};
  virtual void saveCurrentEvent(std::string /*targetFolder*/){};
  virtual int getRunNumber() const { return 0; }
  virtual void setRunNumber(int) {}
  virtual std::string getEventName() { return "event"; };
  virtual std::string getEventAbsoluteFilePath() { return ""; };
  virtual int getFirstTForbit() const { return 0; }
  virtual void setFirstTForbit(int) {}
  virtual std::string getCollisionTime() const { return "not specified"; }
  virtual void setCollisionTime(std::string) {}
  virtual int getTrackMask() const { return 0; }
  virtual void setTrackMask(int) {}
  virtual int getClusterMask() const { return 0; }
  virtual void setClusterMask(int) {}
  virtual o2::detectors::DetID::mask_t getDetectorsMask() = 0;
};

} // namespace event_visualisation
} // namespace o2

#endif // ALICE_O2_EVENTVISUALISATION_BASE_DATASOURCE_H
