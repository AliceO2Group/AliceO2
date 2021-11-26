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
#include <EventVisualisationBase/DataSourceOnline.h>
#include <EventVisualisationBase/FileWatcher.h>

class TObject;

namespace o2
{
namespace event_visualisation
{

class DataSourceOffline : public DataSourceOnline
{
 public:
  DataSourceOffline(const std::string path, const std::string file);

  ~DataSourceOffline() override = default;
  DataSourceOffline(DataSourceOffline const&) = delete;

  /// Deleted assigment operator
  void operator=(DataSourceOffline const&) = delete;
};

} // namespace event_visualisation
} // namespace o2

#endif //O2EVE_DATASOURCEOFFLINE_H
