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

#ifndef O2EVE_DATASOURCEONLINE_H
#define O2EVE_DATASOURCEONLINE_H

#include <EventVisualisationBase/DataSource.h>
#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationBase/FileWatcher.h>
#include <string>
#include <vector>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSourceOnline : public DataSource
{
 protected:
  static std::vector<std::string> sourceFilextensions;
  FileWatcher mFileWatcher;
  int mRunNumber;
  o2::parameters::GRPECS::RunType mRunType;
  int mFirstTForbit;
  int mTrackMask;
  int mClusterMask;
  std::string mCollisionTime;
  std::string mFileTime;

 public:
  DataSourceOnline(const std::string path);

  ~DataSourceOnline() override = default;
  DataSourceOnline(DataSourceOnline const&) = delete;

  /// Deleted assigment operator
  void operator=(DataSourceOnline const&) = delete;

  int getEventCount() override;
  void setCurrentEvent(Int_t currentEvent) override;
  Int_t getCurrentEvent() override;

  bool refresh() override; // recompute

  std::vector<std::pair<VisualisationEvent, EVisualisationGroup>> getVisualisationList(int no, float minTime, float maxTime, float range) override;
  bool rollToNext() override { return mFileWatcher.rollToNext(); };
  void changeDataFolder(std::string newFolder) override { mFileWatcher.changeFolder(newFolder); };
  void saveCurrentEvent(std::string targetFolder) override { mFileWatcher.saveCurrentFileToFolder(targetFolder); };
  int getRunNumber() const override { return this->mRunNumber; }
  void setRunNumber(int runNumber) override { this->mRunNumber = runNumber; }
  parameters::GRPECS::RunType getRunType() override { return mRunType; }
  void setRunType(parameters::GRPECS::RunType runType) override { this->mRunType = runType; }
  std::string getEventName() override { return mFileWatcher.currentItem(); };
  std::string getEventAbsoluteFilePath() override { return mFileWatcher.currentFilePath(); };
  int getFirstTForbit() const override { return this->mFirstTForbit; }
  void setFirstTForbit(int firstTForbit) override { this->mFirstTForbit = firstTForbit; }
  std::string getCollisionTime() const override { return this->mCollisionTime; }
  void setCollisionTime(std::string collisionTime) override { this->mCollisionTime = collisionTime; }
  std::string getFileTime() const override { return this->mFileTime; }
  void setFileTime(std::string fileTime) override { this->mFileTime = fileTime; }
  int getTrackMask() const override { return this->mTrackMask; }
  void setTrackMask(int trackMask) override { this->mTrackMask = trackMask; }
  int getClusterMask() const override { return this->mClusterMask; }
  void setClusterMask(int clusterMask) override { this->mClusterMask = clusterMask; }
  o2::detectors::DetID::mask_t getDetectorsMask() override;
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_DATASOURCEONLINE_H
