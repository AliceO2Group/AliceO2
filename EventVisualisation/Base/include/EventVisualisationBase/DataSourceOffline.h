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

/// \file DataSourceOffline.h
/// \brief Grouping reading from file(s)
/// \author p.nowakowski@cern.ch

#ifndef O2EVE_DATASOURCEOFFLINE_H
#define O2EVE_DATASOURCEOFFLINE_H

#include <EventVisualisationBase/DataSource.h>
#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationBase/FileWatcher.h>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSourceOffline : public DataSource
{
 private:
  Int_t mCurrentEvent;
  std::vector<VisualisationEvent> mEvents;

 public:
  DataSourceOffline();

  ~DataSourceOffline() override = default;
  DataSourceOffline(DataSourceOffline const&) = delete;

  /// Deleted assigment operator
  void operator=(DataSourceOffline const&) = delete;

  Int_t getCurrentEvent() override { return mCurrentEvent; };
  void setCurrentEvent(Int_t currentEvent) override;
  int getEventCount() override { return mEvents.size(); };
  bool refresh() override { return false; }; // recompute

  virtual std::vector<std::pair<VisualisationEvent, EVisualisationGroup>> getVisualisationList(int no) override;

  void addEvent(VisualisationEvent const& event);

  void changeDataFolder(std::string /*newFolder*/) override {};
  void saveCurrentEvent(std::string /*targetFolder*/) override {};
};

} // namespace event_visualisation
} // namespace o2

#endif // O2EVE_DATASOURCEOFFLINE_H
