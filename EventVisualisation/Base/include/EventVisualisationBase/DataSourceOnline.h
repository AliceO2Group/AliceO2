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

#ifndef O2EVE_DATASOURCEONLINE_H
#define O2EVE_DATASOURCEONLINE_H

#include <EventVisualisationBase/DataSource.h>
#include <EventVisualisationBase/DataReader.h>
#include <EventVisualisationBase/FileWatcher.h>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSourceOnline : public DataSource
{
 protected:
  FileWatcher mFileWatcher;

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

  std::vector<std::pair<VisualisationEvent, std::string>> getVisualisationList(int no) override;
};

} // namespace event_visualisation
} // namespace o2

#endif //O2EVE_DATASOURCEONLINE_H
